"""
第05节: 网关与路由 -- "每条消息都能找到归宿"

Gateway 是消息枢纽: 每条入站消息解析为 (agent_id, session_key)。
路由系统是一个五层绑定表, 从最具体到最通用进行匹配。

第04节解决的是“怎么接多个社交媒体平台”:
    Telegram / 飞书 / CLI -> 统一成 InboundMessage

第05节解决的是下一步问题:
    同一套平台入口后面, 可能不止一个 agent。
    那么每条消息进来时, 需要先判断:
        1. 这条消息应该交给哪个 agent?
        2. 这条消息应该接到哪个历史会话 session?

所以本节新增两个核心概念:
    Gateway:
        消息入口/枢纽。它接收 REPL 或 WebSocket 传来的消息请求,
        把 channel、account_id、peer_id 等上下文交给路由系统。

    Routing / BindingTable:
        路由表。它根据 peer_id、guild_id、account_id、channel、default
        这些规则, 决定消息属于哪个 agent。

一句话:
    Channel 解决“从哪个平台来、怎么发回平台”。
    Gateway + Routing 解决“来了以后交给哪个 agent、用哪段上下文”。

    入站消息 (channel, account_id, peer_id, text)
           |
    +------v------+     +----------+
    |   Gateway    | <-- | WS/REPL  |  JSON-RPC 2.0
    +------+------+     +----------+
           |
    +------v------+
    |   Routing    |  5层: peer > guild > account > channel > default
    +------+------+
           |
     (agent_id, session_key)
           |
    +------v------+
    | AgentManager |  每个 agent 的配置 / 工作区 / 会话
    +------+------+
           |
        LLM API

运行方法:  cd claw0 && python zh/s05_gateway_routing.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# 导入 & 配置
# ---------------------------------------------------------------------------
import os, re, sys, json, time, asyncio, threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
AGENTS_DIR = WORKSPACE_DIR / ".agents"

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
BOLD, MAGENTA, RED, BLUE = "\033[1m", "\033[35m", "\033[31m", "\033[34m"
MAX_TOOL_OUTPUT = 30000

# ---------------------------------------------------------------------------
# Agent ID 标准化
# ---------------------------------------------------------------------------

# agent_id 会出现在路由表、目录名、session key 里, 所以需要先标准化。
# 这里限制为小写字母/数字/_/-, 避免出现空格、中文、斜杠等不适合做 id 的字符。
VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
DEFAULT_AGENT_ID = "main"

def normalize_agent_id(value: str) -> str:
    # 例子:
    #   "Sage"       -> "sage"
    #   "my bot!"    -> "my-bot"
    #   ""           -> "main"
    # 这样用户在命令行或 JSON-RPC 里传来的名字不会把系统搞乱。
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    if VALID_ID_RE.match(trimmed):
        return trimmed.lower()
    cleaned = INVALID_CHARS_RE.sub("-", trimmed.lower()).strip("-")[:64]
    return cleaned or DEFAULT_AGENT_ID

# ---------------------------------------------------------------------------
# 绑定: 五层路由解析
# ---------------------------------------------------------------------------
# 这里的 BindingTable 可以理解成“消息分配规则表”。
#
# 一条入站消息至少会带:
#   channel    = 平台, 例如 cli / telegram / discord
#   peer_id    = 对话对象, 例如用户 id、群 id、话题 id
# 还可能带:
#   account_id = 哪个 bot 账号收到的
#   guild_id   = Discord/Slack 这类平台的服务器/团队空间 id
#
# 路由表从最具体到最宽泛匹配:
#   第1层: peer_id    -- 某个具体用户/群/会话指定给某个 agent
#   第2层: guild_id   -- 某个服务器/团队空间指定给某个 agent
#   第3层: account_id -- 某个 bot 账号收到的消息指定给某个 agent
#   第4层: channel    -- 某个平台来的消息都指定给某个 agent
#   第5层: default    -- 兜底, 前面都没命中时使用
#
# 为什么要“越具体越优先”:
#   如果你配置了 “所有 Telegram -> sage”,
#   但又配置了 “telegram:admin-001 -> luna”,
#   那 admin-001 这条更具体的规则应该覆盖整个平台规则。

@dataclass
class Binding:
    # 一条 Binding 就是一条“如果匹配 X, 就路由到 agent Y”的规则。
    agent_id: str
    tier: int           # 1-5, 越小越具体
    match_key: str      # "peer_id" | "guild_id" | "account_id" | "channel" | "default"
    match_value: str    # 例如 "telegram:12345", "discord", "*"
    priority: int = 0   # 同层内, 越大越优先

    def display(self) -> str:
        # 只用于命令行展示, 让 /bindings 输出更容易读。
        names = {1: "peer", 2: "guild", 3: "account", 4: "channel", 5: "default"}
        label = names.get(self.tier, f"tier-{self.tier}")
        return f"[{label}] {self.match_key}={self.match_value} -> agent:{self.agent_id} (pri={self.priority})"

class BindingTable:
    def __init__(self) -> None:
        # 内存版路由表。生产系统通常会从配置文件/数据库加载这些绑定。
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        self._bindings.append(binding)
        # 排序规则:
        #   先按 tier 升序: 1 比 5 更具体, 所以先检查。
        #   再按 priority 降序: 同一层里 priority 大的先检查。
        #
        # resolve() 之后只需要从头扫到尾, 第一个匹配的就是胜者。
        self._bindings.sort(key=lambda b: (b.tier, -b.priority))

    def remove(self, agent_id: str, match_key: str, match_value: str) -> bool:
        # 按 agent_id + match_key + match_value 删除规则。
        # 教学版没有做 tier/priority 的精确删除, 因为 REPL 里只展示核心路由逻辑。
        before = len(self._bindings)
        self._bindings = [
            b for b in self._bindings
            if not (b.agent_id == agent_id and b.match_key == match_key
                    and b.match_value == match_value)
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        return list(self._bindings)

    def resolve(self, channel: str = "", account_id: str = "",
                guild_id: str = "", peer_id: str = "") -> tuple[str | None, Binding | None]:
        """
        遍历第1-5层, 第一个匹配的获胜。返回 (agent_id, matched_binding)。

        这里没有复杂算法, 本质就是:
            for 每条规则:
                如果规则能匹配这条消息:
                    返回规则指向的 agent

        关键是规则已经在 add() 时排序好了, 所以“第一个匹配”天然就是
        “最具体、同层优先级最高”的那条。
        """
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                # peer_id 是最具体的一层。
                #
                # 支持两种写法:
                #   match_value="admin-001"          -> 任何平台的 admin-001
                #   match_value="discord:admin-001"  -> 只匹配 Discord 的 admin-001
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                # guild_id 常见于 Discord/Slack: 一个服务器/工作区下有多个频道和用户。
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                # account_id 是“我方 bot 账号”。同一平台多个 bot 可以路由给不同 agent。
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                # channel 级别规则: 某个平台来的所有消息都交给一个 agent。
                # 例如所有 telegram 消息默认给 sage。
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                # 兜底规则。只要前面没有命中, 就让 default 接住消息。
                return b.agent_id, b
        return None, None

# ---------------------------------------------------------------------------
# 会话键构建
# ---------------------------------------------------------------------------
# route 决定“给哪个 agent”。
# session_key 决定“用这个 agent 的哪段历史对话”。
#
# 这两个概念不要混在一起:
#   agent_id     = 谁来回答, 例如 luna / sage
#   session_key  = 这次对话接到哪段上下文, 例如 agent:luna:direct:user-001
#
# 同一个 agent 可以同时服务很多人, 每个人应该有自己的历史上下文;
# 否则 A 用户刚说的隐私内容, B 用户下一句可能就接上了。
#
# dm_scope 控制私聊隔离粒度:
#   main                      -> agent:{id}:main
#   per-peer                  -> agent:{id}:direct:{peer}
#   per-channel-peer          -> agent:{id}:{ch}:direct:{peer}
#   per-account-channel-peer  -> agent:{id}:{ch}:{acc}:direct:{peer}

def build_session_key(agent_id: str, channel: str = "", account_id: str = "",
                      peer_id: str = "", dm_scope: str = "per-peer") -> str:
    # 先把参与 key 的部分标准化, 避免 "Telegram" 和 "telegram" 被当成两段会话。
    aid = normalize_agent_id(agent_id)
    ch = (channel or "unknown").strip().lower()
    acc = (account_id or "default").strip().lower()
    pid = (peer_id or "").strip().lower()
    if dm_scope == "per-account-channel-peer" and pid:
        # 最大隔离度:
        # 同一个用户在不同平台、不同 bot 账号下都会有不同上下文。
        # 适合生产环境或多租户机器人。
        return f"agent:{aid}:{ch}:{acc}:direct:{pid}"
    if dm_scope == "per-channel-peer" and pid:
        # 中等隔离度:
        # 同一个 peer_id 如果同时出现在 Telegram 和 Discord, 会分成两段上下文。
        return f"agent:{aid}:{ch}:direct:{pid}"
    if dm_scope == "per-peer" and pid:
        # 最常用的教学默认:
        # 每个 peer_id 一段上下文, 但不区分平台和 bot 账号。
        return f"agent:{aid}:direct:{pid}"
    # main 表示所有人共享同一段上下文。适合单用户 demo, 不适合真实多用户聊天。
    return f"agent:{aid}:main"

# ---------------------------------------------------------------------------
# Agent 配置 & 管理器
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """
    一个 agent 的配置。

    第04节只有一个默认大脑; 第05节开始有多个 agent。
    每个 agent 可以有自己的名字、性格、模型和会话隔离策略。
    """
    id: str
    name: str
    personality: str = ""
    model: str = ""
    dm_scope: str = "per-peer"

    @property
    def effective_model(self) -> str:
        # 如果 agent 没有单独指定模型, 就使用全局 MODEL_ID。
        return self.model or MODEL_ID

    def system_prompt(self) -> str:
        # 这里把 AgentConfig 转成 LLM 的 system prompt。
        # 路由选中不同 agent 后, 这里生成的提示词也会不同,
        # 所以 luna 和 sage 会表现出不同风格。
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)

class AgentManager:
    def __init__(self, agents_base: Path | None = None) -> None:
        # agent_id -> AgentConfig
        self._agents: dict[str, AgentConfig] = {}
        # 每个 agent 的本地目录根路径。教学版只创建目录, 不做完整持久化。
        self._agents_base = agents_base or AGENTS_DIR
        # session_key -> Anthropic messages
        # 这是内存里的对话历史。session_key 不同, 历史上下文就不同。
        self._sessions: dict[str, list[dict]] = {}

    def register(self, config: AgentConfig) -> None:
        # 注册 agent 时标准化 id, 并创建对应目录。
        # 目录结构让你看到生产系统通常会把 agent 配置、会话、工作区拆开管理。
        aid = normalize_agent_id(config.id)
        config.id = aid
        self._agents[aid] = config
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKSPACE_DIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[AgentConfig]:
        return list(self._agents.values())

    def get_session(self, session_key: str) -> list[dict]:
        # LLM API 的 messages 就存在这里。
        # 如果这是该 session_key 第一次出现, 就创建一段新的空历史。
        if session_key not in self._sessions:
            self._sessions[session_key] = []
        return self._sessions[session_key]

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        # 返回每个 session 当前有多少条 messages, 给 /sessions 和 Gateway API 使用。
        aid = normalize_agent_id(agent_id) if agent_id else ""
        return {k: len(v) for k, v in self._sessions.items()
                if not aid or k.startswith(f"agent:{aid}:")}

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

# 第05节的重点不是工具系统, 但保留两个简单工具, 让多个 agent 都能走完整的
# “LLM -> tool_use -> tool_result -> LLM 回复”循环。
TOOLS = [
    {"name": "read_file", "description": "Read the contents of a file.",
     "input_schema": {"type": "object", "required": ["file_path"],
                      "properties": {"file_path": {"type": "string", "description": "Path to the file."}}}},
    {"name": "get_current_time", "description": "Get the current date and time in UTC.",
     "input_schema": {"type": "object", "properties": {}}},
]

def _tool_read(file_path: str) -> str:
    # 简化版读文件工具。真实系统会做更严格的工作区隔离和权限检查。
    try:
        p = Path(file_path).resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except Exception as exc:
        return f"Error: {exc}"

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": lambda file_path: _tool_read(file_path),
    "get_current_time": lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
}

def process_tool_call(name: str, inp: dict) -> str:
    # 把模型发出的 tool_use 分发给本地 Python 函数。
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Error: Unknown tool '{name}'"
    try:
        return handler(**inp)
    except Exception as exc:
        return f"Error: {name} failed: {exc}"

# ---------------------------------------------------------------------------
# 共享事件循环 (持久化后台线程)
# ---------------------------------------------------------------------------

# GatewayServer 是 async WebSocket 服务, 但 REPL 是同步 input() 循环。
# 为了让两者共存, 这里在后台线程里启动一个长期运行的 asyncio event loop。
# REPL 需要跑 async agent 时, 用 run_async() 把 coroutine 丢到这个 loop 里执行。
_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

def get_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop, _loop_thread
    if _event_loop is not None and _event_loop.is_running():
        return _event_loop
    _event_loop = asyncio.new_event_loop()
    def _run():
        # 后台线程绑定这个 event loop, 然后一直 run_forever()。
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()
    _loop_thread = threading.Thread(target=_run, daemon=True)
    _loop_thread.start()
    return _event_loop

def run_async(coro):
    # 从同步 REPL 里等待异步任务结果。
    # 例如 reply = run_async(run_agent(...))。
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

# ---------------------------------------------------------------------------
# 路由解析
# ---------------------------------------------------------------------------

def resolve_route(bindings: BindingTable, mgr: AgentManager,
                  channel: str, peer_id: str,
                  account_id: str = "", guild_id: str = "") -> tuple[str, str]:
    """
    把一条入站消息解析成 (agent_id, session_key)。

    输入来自第04节里的统一消息上下文:
        channel    消息来自哪个平台
        peer_id    消息来自哪个用户/群/会话
        account_id 哪个 bot 账号收到消息
        guild_id   哪个服务器/工作区

    输出给 agent runner:
        agent_id    由哪个 agent 回答
        session_key 使用哪段历史上下文
    """
    agent_id, matched = bindings.resolve(
        channel=channel, account_id=account_id,
        guild_id=guild_id, peer_id=peer_id,
    )
    if not agent_id:
        # 没有任何绑定命中时, 落到 main。
        # demo 的 setup_demo() 默认会注册 default 绑定, 所以一般不会走到这里。
        agent_id = DEFAULT_AGENT_ID
        print(f"  {DIM}[route] No binding matched, default: {agent_id}{RESET}")
    elif matched:
        print(f"  {DIM}[route] Matched: {matched.display()}{RESET}")
    agent = mgr.get_agent(agent_id)
    # agent 选出来以后, 再看这个 agent 的 dm_scope, 决定 session 隔离粒度。
    # 这一步非常重要: 路由表只决定“谁来回答”, dm_scope 决定“接哪段上下文”。
    dm_scope = agent.dm_scope if agent else "per-peer"
    sk = build_session_key(agent_id, channel=channel, account_id=account_id,
                           peer_id=peer_id, dm_scope=dm_scope)
    return agent_id, sk

# ---------------------------------------------------------------------------
# Agent 运行器
# ---------------------------------------------------------------------------

# 全局并发限制。多个 WebSocket 客户端同时发消息时, 最多同时跑 4 个 agent turn。
_agent_semaphore: asyncio.Semaphore | None = None

async def run_agent(mgr: AgentManager, agent_id: str, session_key: str,
                    user_text: str, on_typing: Any = None) -> str:
    """
    执行一次 agent 回合。

    到这里时, Gateway/路由已经完成:
        agent_id    已经知道由哪个 agent 回答
        session_key 已经知道使用哪段历史

    run_agent() 不再关心消息来自 CLI、Telegram 还是 WebSocket。
    它只根据 agent_id 取配置, 根据 session_key 取 messages, 然后调用 LLM。
    """
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    # 取出这段会话历史。后续 messages.append() 会直接修改同一个 list,
    # 所以这段 session 会持续记住上下文。
    messages = mgr.get_session(session_key)
    messages.append({"role": "user", "content": user_text})
    async with _agent_semaphore:
        if on_typing:
            # Gateway 用这个回调广播 typing 状态, REPL 不传就没有 typing 事件。
            on_typing(agent_id, True)
        try:
            return await _agent_loop(agent.effective_model, agent.system_prompt(), messages)
        finally:
            if on_typing:
                on_typing(agent_id, False)

async def _agent_loop(model: str, system: str, messages: list[dict]) -> str:
    # 标准工具循环:
    #   1. 带着 messages 调模型
    #   2. 如果模型结束, 返回文本
    #   3. 如果模型请求工具, 本地执行工具并把 tool_result 追加回 messages
    #   4. 继续调用模型, 直到 end_turn 或达到迭代上限
    for _ in range(15):
        try:
            # Anthropic client 是同步 SDK, 放到线程里跑, 避免阻塞 asyncio event loop。
            response = await asyncio.to_thread(
                client.messages.create,
                model=model, max_tokens=4096,
                system=system, tools=TOOLS, messages=messages,
            )
        except Exception as exc:
            # API 失败时回滚本轮追加的消息, 避免坏掉的 assistant/tool 片段污染会话历史。
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            # 正常完成: 把 assistant 文本块拼起来返回给 Gateway/REPL。
            return "".join(b.text for b in response.content if hasattr(b, "text")) or "[no text]"
        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  {DIM}[tool: {block.name}]{RESET}")
                # 模型请求的工具结果必须以 tool_result 形式追加成下一条 user message。
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": process_tool_call(block.name, block.input)})
            messages.append({"role": "user", "content": results})
            continue
        return "".join(b.text for b in response.content if hasattr(b, "text")) or f"[stop={response.stop_reason}]"
    return "[max iterations reached]"

# ---------------------------------------------------------------------------
# Gateway 服务器 (WebSocket, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

# GatewayServer 是“远程入口”示例。
#
# 第04节的 TelegramChannel 直接从 Telegram 拉消息;
# 第05节这里换成 WebSocket + JSON-RPC:
#   外部程序连到 ws://localhost:8765
#   发送 {"method": "send", "params": {"channel": "...", "peer_id": "...", "text": "..."}}
#   Gateway 根据 channel/peer_id 路由到 agent, 跑完后返回 reply。
#
# 所以 Gateway 不等于某个社交平台, 它更像平台适配层后面的统一入口:
#   Telegram adapter  \
#   Feishu adapter     -> Gateway -> Routing -> AgentManager -> LLM
#   WebSocket client  /
class GatewayServer:
    def __init__(self, mgr: AgentManager, bindings: BindingTable,
                 host: str = "localhost", port: int = 8765) -> None:
        # Gateway 持有同一份 AgentManager 和 BindingTable。
        # REPL 和 WebSocket 共享它们, 所以在 REPL 里新增绑定后, Gateway 也能看到。
        self._mgr = mgr
        self._bindings = bindings
        self._host, self._port = host, port
        # 当前连接的 WebSocket 客户端集合。typing 事件会广播给这些客户端。
        self._clients: set[Any] = set()
        self._start_time = time.monotonic()
        self._server: Any = None
        self._running = False

    async def start(self) -> None:
        try:
            import websockets
        except ImportError:
            print(f"{RED}websockets not installed. pip install websockets{RESET}"); return
        self._start_time = time.monotonic()
        self._running = True
        # websockets.serve() 会把每个连接交给 _handle()。
        self._server = await websockets.serve(self._handle, self._host, self._port)
        print(f"{GREEN}Gateway started ws://{self._host}:{self._port}{RESET}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False

    async def _handle(self, ws: Any, path: str = "") -> None:
        self._clients.add(ws)
        try:
            async for raw in ws:
                # 每收到一条 WebSocket 文本消息, 就按 JSON-RPC 请求处理。
                resp = await self._dispatch(raw)
                if resp:
                    await ws.send(json.dumps(resp))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def _typing_cb(self, agent_id: str, typing: bool) -> None:
        # run_agent() 开始/结束时会调用这个回调。
        # Gateway 把它包装成 JSON-RPC notification 广播给所有连接客户端。
        msg = json.dumps({"jsonrpc": "2.0", "method": "typing",
                          "params": {"agent_id": agent_id, "typing": typing}})
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(msg))
            except Exception:
                self._clients.discard(ws)

    async def _dispatch(self, raw: str) -> dict | None:
        # 最小 JSON-RPC 2.0 分发器。
        # 请求格式示例:
        #   {"jsonrpc":"2.0","id":1,"method":"send","params":{"text":"hi"}}
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params", {})
        # 对外暴露的方法表。每个 method 映射到一个 _m_xxx handler。
        methods = {
            "send": self._m_send, "bindings.set": self._m_bind_set,
            "bindings.list": self._m_bind_list, "sessions.list": self._m_sessions,
            "agents.list": self._m_agents, "status": self._m_status,
        }
        handler = methods.get(method)
        if not handler:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown: {method}"}, "id": rid}
        try:
            return {"jsonrpc": "2.0", "result": await handler(params), "id": rid}
        except Exception as exc:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": rid}

    async def _m_send(self, p: dict) -> dict:
        """
        Gateway 最核心的方法: 收到一条消息, 返回 agent 回复。

        params 可以长这样:
            {
              "channel": "telegram",
              "peer_id": "12345",
              "account_id": "tg-primary",
              "text": "hello"
            }

        这和第04节 InboundMessage 的字段是同一套概念。真实平台适配器可以把
        InboundMessage 转成这个 JSON-RPC 请求, 也可以在同进程里直接调用 resolve_route()。
        """
        text = p.get("text", "")
        if not text:
            raise ValueError("text is required")
        ch, pid = p.get("channel", "websocket"), p.get("peer_id", "ws-client")
        acc, gid = p.get("account_id", ""), p.get("guild_id", "")
        if p.get("agent_id"):
            # 显式指定 agent_id 时, 跳过路由表。
            # 这类似 REPL 里的 /switch, 适合调试或管理端强制指定。
            aid = normalize_agent_id(p["agent_id"])
            a = self._mgr.get_agent(aid)
            sk = build_session_key(aid, channel=ch, account_id=acc, peer_id=pid,
                                   dm_scope=a.dm_scope if a else "per-peer")
        else:
            # 普通路径: 根据 channel/peer_id/account_id/guild_id 查 BindingTable,
            # 得到 agent_id 和 session_key。
            aid, sk = resolve_route(self._bindings, self._mgr, ch, pid,
                                    account_id=acc, guild_id=gid)
        # 交给 agent runner。它会取 agent 配置、取 session 历史、调用 LLM。
        reply = await run_agent(self._mgr, aid, sk, text, on_typing=self._typing_cb)
        return {"agent_id": aid, "session_key": sk, "reply": reply}

    async def _m_bind_set(self, p: dict) -> dict:
        # 远程新增路由规则。
        # 例如把所有 telegram 消息改到 sage:
        #   method="bindings.set",
        #   params={"agent_id":"sage","tier":4,"match_key":"channel","match_value":"telegram"}
        b = Binding(agent_id=normalize_agent_id(p.get("agent_id", "")),
                    tier=int(p.get("tier", 5)), match_key=p.get("match_key", "default"),
                    match_value=p.get("match_value", "*"), priority=int(p.get("priority", 0)))
        self._bindings.add(b)
        return {"ok": True, "binding": b.display()}

    async def _m_bind_list(self, p: dict) -> list[dict]:
        # 返回当前路由表, 给 UI/调试客户端展示。
        return [{"agent_id": b.agent_id, "tier": b.tier, "match_key": b.match_key,
                 "match_value": b.match_value, "priority": b.priority}
                for b in self._bindings.list_all()]

    async def _m_sessions(self, p: dict) -> dict:
        # 返回当前内存里的 session 列表和消息数量。
        return self._mgr.list_sessions(p.get("agent_id", ""))

    async def _m_agents(self, p: dict) -> list[dict]:
        # 返回可用 agent 及其配置。
        return [{"id": a.id, "name": a.name, "model": a.effective_model,
                 "dm_scope": a.dm_scope, "personality": a.personality}
                for a in self._mgr.list_agents()]

    async def _m_status(self, p: dict) -> dict:
        # 健康检查/状态接口。
        return {"running": self._running,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "connected_clients": len(self._clients),
                "agent_count": len(self._mgr.list_agents()),
                "binding_count": len(self._bindings.list_all())}

# ---------------------------------------------------------------------------
# 演示: 双 agent (luna + sage) + 路由绑定
# ---------------------------------------------------------------------------

def setup_demo() -> tuple[AgentManager, BindingTable]:
    # 创建两个不同风格的 agent, 用来演示“同一套入口后面可以接多个大脑”。
    mgr = AgentManager()
    mgr.register(AgentConfig(
        id="luna", name="Luna",
        personality="warm, curious, and encouraging. You love asking follow-up questions.",
    ))
    mgr.register(AgentConfig(
        id="sage", name="Sage",
        personality="direct, analytical, and concise. You prefer facts over opinions.",
    ))
    bt = BindingTable()
    # 兜底: 什么都没匹配到时, 交给 luna。
    bt.add(Binding(agent_id="luna", tier=5, match_key="default", match_value="*"))
    # 平台级: 所有 Telegram 消息交给 sage。
    # 这会覆盖 default, 因为 tier=4 比 tier=5 更具体。
    bt.add(Binding(agent_id="sage", tier=4, match_key="channel", match_value="telegram"))
    # 具体 peer 级: Discord 里的 admin-001 交给 sage。
    # 这会覆盖 channel/default, 因为 tier=1 最具体。
    bt.add(Binding(agent_id="sage", tier=1, match_key="peer_id",
                   match_value="discord:admin-001", priority=10))
    return mgr, bt

# ---------------------------------------------------------------------------
# REPL + 命令
# ---------------------------------------------------------------------------

def cmd_bindings(bt: BindingTable) -> None:
    # 查看当前路由表。颜色按 tier 深浅区分, 只是为了命令行易读。
    all_b = bt.list_all()
    if not all_b:
        print(f"  {DIM}(no bindings){RESET}"); return
    print(f"\n{BOLD}Route Bindings ({len(all_b)}):{RESET}")
    for b in all_b:
        c = [MAGENTA, BLUE, CYAN, GREEN, DIM][min(b.tier - 1, 4)]
        print(f"  {c}{b.display()}{RESET}")
    print()

def cmd_route(bt: BindingTable, mgr: AgentManager, args: str) -> None:
    # 手动模拟一条入站消息, 看它会被路由到哪个 agent、哪个 session。
    # 用法:
    #   /route cli user1
    #   /route telegram user2
    #   /route discord admin-001
    parts = args.strip().split()
    if len(parts) < 2:
        print(f"  {YELLOW}Usage: /route <channel> <peer_id> [account_id] [guild_id]{RESET}"); return
    ch, pid = parts[0], parts[1]
    acc = parts[2] if len(parts) > 2 else ""
    gid = parts[3] if len(parts) > 3 else ""
    aid, sk = resolve_route(bt, mgr, channel=ch, peer_id=pid, account_id=acc, guild_id=gid)
    a = mgr.get_agent(aid)
    print(f"\n{BOLD}Route Resolution:{RESET}")
    print(f"  {DIM}Input:   ch={ch} peer={pid} acc={acc or '-'} guild={gid or '-'}{RESET}")
    print(f"  {CYAN}Agent:   {aid} ({a.name if a else '?'}){RESET}")
    print(f"  {GREEN}Session: {sk}{RESET}\n")

def cmd_agents(mgr: AgentManager) -> None:
    # 查看当前注册的 agent。这里能看到每个 agent 的 model 和 dm_scope。
    agents = mgr.list_agents()
    if not agents:
        print(f"  {DIM}(no agents){RESET}"); return
    print(f"\n{BOLD}Agents ({len(agents)}):{RESET}")
    for a in agents:
        print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}  dm_scope={a.dm_scope}")
        if a.personality:
            print(f"    {DIM}{a.personality[:70]}{'...' if len(a.personality) > 70 else ''}{RESET}")
    print()

def cmd_sessions(mgr: AgentManager) -> None:
    # 查看已出现过的会话。你和不同 agent/peer 对话后, 这里会出现不同 session_key。
    s = mgr.list_sessions()
    if not s:
        print(f"  {DIM}(no sessions){RESET}"); return
    print(f"\n{BOLD}Sessions ({len(s)}):{RESET}")
    for k, n in sorted(s.items()):
        print(f"  {GREEN}{k}{RESET} ({n} msgs)")
    print()

def repl() -> None:
    # REPL 是本节的本地演示入口。
    # 它模拟一条来自 channel="cli", peer_id="repl-user" 的消息,
    # 然后走和 Gateway 一样的 resolve_route() + run_agent() 流程。
    mgr, bindings = setup_demo()
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  claw0  |  Section 05: Gateway & Routing{RESET}")
    print(f"{DIM}  Model: {MODEL_ID}{RESET}")
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  /bindings  /route <ch> <peer>  /agents  /sessions  /switch <id>  /gateway{RESET}")
    print()

    ch, pid = "cli", "repl-user"
    # force_agent 非空时, 跳过路由表, 强制把消息交给指定 agent。
    # 这就是 /switch sage 的效果。
    force_agent = ""
    gw_started = False

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}"); break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}"); break

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input[len(cmd):].strip()
            if cmd == "/bindings":
                cmd_bindings(bindings)
            elif cmd == "/route":
                cmd_route(bindings, mgr, args)
            elif cmd == "/agents":
                cmd_agents(mgr)
            elif cmd == "/sessions":
                cmd_sessions(mgr)
            elif cmd == "/switch":
                if not args:
                    print(f"  {DIM}force={force_agent or '(off)'}{RESET}")
                elif args.lower() == "off":
                    # 恢复正常路由: 后续消息重新交给 BindingTable 判断。
                    force_agent = ""
                    print(f"  {DIM}Routing mode restored.{RESET}")
                else:
                    aid = normalize_agent_id(args)
                    if mgr.get_agent(aid):
                        # 强制模式: 之后的普通输入都直接给这个 agent。
                        force_agent = aid
                        print(f"  {GREEN}Forcing: {aid}{RESET}")
                    else:
                        print(f"  {YELLOW}Not found: {aid}{RESET}")
            elif cmd == "/gateway":
                if gw_started:
                    print(f"  {DIM}Already running.{RESET}")
                else:
                    # 在后台 event loop 启动 WebSocket Gateway。
                    # 启动后, 外部客户端就可以通过 JSON-RPC 调同一套路由和 agent 逻辑。
                    gw = GatewayServer(mgr, bindings)
                    asyncio.run_coroutine_threadsafe(gw.start(), get_event_loop())
                    print(f"{GREEN}Gateway running in background on ws://localhost:8765{RESET}\n")
                    gw_started = True
            else:
                print(f"  {YELLOW}Unknown: {cmd}{RESET}")
            continue

        if force_agent:
            # 强制指定 agent: 不查 BindingTable, 但仍然要根据该 agent 的 dm_scope
            # 构造 session_key, 否则上下文不知道放在哪里。
            agent_id = force_agent
            a = mgr.get_agent(agent_id)
            session_key = build_session_key(agent_id, channel=ch, peer_id=pid,
                                            dm_scope=a.dm_scope if a else "per-peer")
        else:
            # 普通路径: CLI 消息也走路由。默认 demo 里 cli 会命中 default -> luna。
            agent_id, session_key = resolve_route(bindings, mgr, channel=ch, peer_id=pid)

        agent = mgr.get_agent(agent_id)
        name = agent.name if agent else agent_id
        print(f"  {DIM}-> {name} ({agent_id}) | {session_key}{RESET}")

        try:
            reply = run_async(run_agent(mgr, agent_id, session_key, user_input))
        except Exception as exc:
            print(f"\n{RED}Error: {exc}{RESET}\n"); continue
        print(f"\n{GREEN}{BOLD}{name}:{RESET} {reply}\n")

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    repl()

if __name__ == "__main__":
    main()
"""

 这一节解决的核心问题是：多个平台进来的消息，怎么交给不同的 agent，并且接到正确的上下文？

  第04节解决的是"怎么接多个平台"，到了第05节，问题升级了——同一套入口后面可能不止一个 agent，那每条消息进来需要先回答两个问题：

  1. 这条消息应该交给哪个 agent？（路由问题）
  2. 这条消息应该接到哪段历史对话？（会话问题）

  ---
  三个核心概念
  
  1. BindingTable（路由表）—— 谁来回答

  五层优先级，从最具体到最宽泛匹配：

  第1层 peer_id     → 某个具体用户指定给某个 agent
  第2层 guild_id    → 某个服务器/工作区指定给某个 agent
  第3层 account_id  → 某个 bot 账号指定给某个 agent
  第4层 channel     → 某个平台指定给某个 agent
  第5层 default     → 兜底

  举例：配置了"所有 Telegram → sage"，但又单独配了"Telegram 的 admin-001 → luna"，那 admin-001 这条更具体的规则会覆盖平台级规则。

  2. dm_scope（会话隔离）—— 接哪段上下文

  路由选完 agent 之后，还要决定这段对话接到哪个 session。dm_scope 控制隔离粒度——是所有人共享一段历史，还是按用户/平台/bot账号隔离。

  3. Gateway（网关）—— 消息枢纽

  统一入口。不管是 REPL 输入还是 WebSocket 连接，消息都先进 Gateway，再交给路由系统。它对外暴露 JSON-RPC 2.0 接口，远程客户端可以通过 WebSocket 调用 send、bindings.set、agents.list 等方法。

  ---
  整条数据流
  
  消息进来 (channel, peer_id, text)
         │
         ▼
     Gateway ── 接收 REPL 或 WebSocket 的消息
         │
         ▼
    BindingTable ── 五层匹配，选出 agent_id
         │
         ▼
    dm_scope ── 根据该 agent 的隔离策略，构造 session_key
         │
         ▼
    AgentManager ── 取出 agent 配置 + 对应 session 的历史消息
         │
         ▼
      LLM API ── 调用模型，得到回复

  一句话总结：第04节让 agent 能从多个平台收发消息，第05节让多个 agent 共存，每条消息都能找到属于自己的 agent 和上下文。


"""