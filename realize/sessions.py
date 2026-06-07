# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import os
import sys
import json              # JSON 序列化/反序列化 (读写 JSONL 和 index)
import uuid              # 生成唯一 session ID
import time              # 时间戳 (记录每条消息的保存时间)
from pathlib import Path
from datetime import datetime, timezone  # UTC 时间 (session 元数据)
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

# 工作空间目录 -- 所有文件操作都限制在此目录下
# 和 s02 的 WORKDIR 不同, 这里用固定的 workspace 子目录
# 这样不同 session 的文件操作不会污染项目根目录
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

# 上下文安全上限 (token 数)
# Claude 的上下文窗口是 200K tokens, 留一点余量设为 180K
# 超过此限制, ContextGuard 会触发压缩机制
CONTEXT_SAFE_LIMIT = 180000

MAX_TOOL_OUTPUT = 50000


# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"     # 紫色 -- 用于 session 相关信息


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_warn(text: str) -> None:
    print(f"{YELLOW}{text}{RESET}")


def print_session(text: str) -> None:
    """打印 session 相关信息, 用紫色区分."""
    print(f"{MAGENTA}{text}{RESET}")


# ---------------------------------------------------------------------------
# 安全路径辅助函数
# ---------------------------------------------------------------------------
def safe_path(raw: str) -> Path:
    """解析路径, 阻止逃逸到 WORKSPACE_DIR 之外。"""
    target = (WORKSPACE_DIR / raw).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target


# ---------------------------------------------------------------------------
# SessionStore -- 基于 JSONL 的会话持久化
# ---------------------------------------------------------------------------
# 为什么需要会话持久化?
#   s01/s02 的 agent 一退出, 所有对话就丢了。
#   实际应用中, 我们需要:
#     - 关掉终端后重新打开, 能接着聊
#     - 切换不同的会话 (比如一个写代码, 一个写文档)
#     - 回顾历史对话
#
# 为什么用 JSONL 而不是数据库?
#   - JSONL = JSON Lines, 每行一个 JSON 对象
#   - 追加写入只需 open("a"), 不需要数据库
#   - 读取时逐行解析, 天然流式
#   - 人类可读, 方便调试
#   - 对 agent 这种 "追加为主, 偶尔回放" 的场景非常合适
#
# 存储结构:
#   workspace/.sessions/agents/<agent_id>/sessions/
#     ├── sessions.json          ← 索引文件 (所有 session 的元数据)
#     ├── <session_id_1>.jsonl   ← session 1 的对话记录
#     ├── <session_id_2>.jsonl   ← session 2 的对话记录
#     └── ...
#
# ┌──────────────────────────────────────────────────────────────┐
# │  SessionStore 的两种数据格式:                                 │
# │                                                              │
# │  JSONL 存储格式 (磁盘上):          API 消息格式 (内存中):     │
# │                                                              │
# │  {"type":"user","content":".."}    {"role":"user",            │
# │                                       "content":".."}        │
# │                                                              │
# │  {"type":"assistant",              {"role":"assistant",       │
# │   "content":[...]}                  "content":[...]}         │
# │                                                              │
# │  {"type":"tool_use",              ← 合并进上面的 assistant   │
# │   "tool_use_id":"..",                消息的 content 列表中    │
# │   "name":"..",                                                │
# │   "input":{..}}                                               │
# │                                                              │
# │  {"type":"tool_result",           ← 合并进 user 消息的       │
# │   "tool_use_id":"..",                content 列表中           │
# │   "content":".."}                                             │
# │                                                              │
# │  写入时拆开 (方便追加)             读取时合并 (API 要求)      │
# └──────────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------


class SessionStore:
    """管理 agent 会话的持久化存储。"""

    def __init__(self, agent_id: str = "default"):
        """
        参数:
            agent_id: agent 标识, 不同 agent 的会话互不干扰
                      比如 "claw0" 和 "claw1" 各有各的 session 目录

        目录结构:
            workspace/.sessions/agents/<agent_id>/sessions/
                ├── sessions.json           ← 索引
                ├── abc123.jsonl            ← 某个 session
                └── def456.jsonl            ← 另一个 session
        """
        self.agent_id = agent_id
        self.base_dir = WORKSPACE_DIR / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # 索引文件: 记录所有 session 的元数据 (label, 创建时间, 消息数等)
        self.index_path = self.base_dir.parent / "sessions.json"
        # 从磁盘加载索引到内存
        self._index: dict[str, dict] = self._load_index()
        # 当前活跃的 session ID
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        """从磁盘加载 session 索引。索引损坏则返回空字典。"""
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        """将内存中的索引写回磁盘。"""
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        """获取某个 session 的 JSONL 文件路径。"""
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        """
        创建新会话。

        流程:
            1. 用 uuid 生成唯一 session ID (取前 12 位, 足够短且不重复)
            2. 在索引中记录元数据 (label, 创建时间等)
            3. 创建空的 JSONL 文件
            4. 设为当前 session

        参数:
            label: 可选的会话标签, 比如 "写代码", "调试bug"

        返回:
            新创建的 session ID
        """
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict]:
        """
        从 JSONL 重建 API 格式的 messages[]。

        这是最关键的方法! 它把磁盘上的 JSONL 行重新组装成
        Anthropic API 要求的消息格式。

        为什么需要 "重建" 而不是直接存 API 格式?
        因为 API 要求消息必须 user/assistant 交替, 且:
          - tool_use 块必须属于 assistant 消息
          - tool_result 块必须属于 user 消息
        但写入时我们是一行行追加的, 工具调用和结果是分开写的。
        所以读取时需要把它们重新 "合并" 到正确的消息中。
        """
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        """
        保存一轮对话 (user 或 assistant) 到当前 session 的 JSONL 文件。

        每次调用追加一行 JSON, 格式如:
          {"type": "user", "content": "帮我看看文件", "ts": 1717700000.0}
          {"type": "assistant", "content": [{"type":"text","text":"好的..."}], "ts": 1717700001.0}
        """
        if not self.current_session_id:
            return
        self.append_transcript(self.current_session_id, {
            "type": role,
            "content": content,
            "ts": time.time(),
        })

    def save_tool_result(self, tool_use_id: str, name: str,
                         tool_input: dict, result: str) -> None:
        """
        保存工具调用的请求和结果。

        写入两行 JSONL:
          1. tool_use 行: 记录模型请求调用什么工具、参数是什么
          2. tool_result 行: 记录工具执行后的返回值

        这两行在 JSONL 中是独立的, 但 _rebuild_history 读取时
        会把 tool_use 合并进 assistant 消息, tool_result 合并进 user 消息。

        参数:
            tool_use_id: 工具调用的唯一 ID (来自模型的 ToolUseBlock.id)
            name:        工具名
            tool_input:  工具参数
            result:      工具执行结果
        """
        if not self.current_session_id:
            return
        ts = time.time()
        # 写入 tool_use 记录
        self.append_transcript(self.current_session_id, {
            "type": "tool_use",
            "tool_use_id": tool_use_id,
            "name": name,
            "input": tool_input,
            "ts": ts,
        })
        # 写入 tool_result 记录
        self.append_transcript(self.current_session_id, {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "ts": ts,
        })

    def append_transcript(self, session_id: str, record: dict) -> None:
        """
        向 JSONL 文件追加一条记录。

        这是最底层的写入方法, 其他 save_* 方法都调用它。
        使用 open("a") 追加模式, 不会覆盖已有内容。
        同时更新索引中的 last_active 和 message_count。
        """
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    def _rebuild_history(self, path: Path) -> list[dict]:
        """
        从 JSONL 行重建 API 格式的消息列表。

        ★ 这是最复杂的方法, 理解它就理解了 JSONL 存储和 API 格式的映射 ★

        Anthropic API 规则决定了这种重建方式:
          - 消息必须 user/assistant 交替
          - tool_use 块属于 assistant 消息
          - tool_result 块属于 user 消息

        JSONL 中每行是独立的, 但 API 要求合并。举例:

          JSONL 行:                              重建后的 API messages:
          ─────────────────────────              ──────────────────────
          {"type":"user","content":"读文件"}  →  {"role":"user","content":"读文件"}

          {"type":"assistant","content":[...]}→  {"role":"assistant","content":[
          {"type":"tool_use","name":"read",...}      TextBlock("让我看看..."),
                                                 ToolUseBlock("read_file",...)
                                               ]}  ← tool_use 合并进 assistant!

          {"type":"tool_result",...}          →  {"role":"user","content":[
                                                 tool_result{...}
                                               ]}  ← tool_result 放进 user!

        合并逻辑:
          - tool_use: 如果上一条是 assistant, 追加到它的 content 列表;
                      否则创建新的 assistant 消息
          - tool_result: 如果上一条是 user 且包含 tool_result, 追加;
                         否则创建新的 user 消息
        """
        messages: list[dict] = []
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == "user":
                # 直接创建 user 消息
                messages.append({
                    "role": "user",
                    "content": record["content"],
                })

            elif rtype == "assistant":
                # assistant 消息的 content 可能是字符串或列表
                # 如果是字符串, 需要转换为 API 要求的 content block 格式
                content = record["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            elif rtype == "tool_use":
                # ★ 关键: tool_use 不是独立消息, 而是合并进 assistant 消息 ★
                # 模型的一次回复可能包含: [TextBlock, ToolUseBlock, ToolUseBlock]
                # JSONL 里分开写了, 读取时要拼回去
                block = {
                    "type": "tool_use",
                    "id": record["tool_use_id"],
                    "name": record["name"],
                    "input": record["input"],
                }
                if messages and messages[-1]["role"] == "assistant":
                    # 上一条是 assistant, 把 tool_use 块追加进去
                    content = messages[-1]["content"]
                    if isinstance(content, list):
                        content.append(block)
                    else:
                        # content 是字符串, 需要先转成列表再追加
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(content)},
                            block,
                        ]
                else:
                    # 没有 assistant 消息, 创建一个新的
                    # (理论上不应发生, 但做防御性处理)
                    messages.append({
                        "role": "assistant",
                        "content": [block],
                    })

            elif rtype == "tool_result":
                # ★ 关键: tool_result 不是独立消息, 合并进 user 消息 ★
                # Anthropic API 要求 tool_result 放在 role="user" 的消息中
                # 多个 tool_result 可以合并到同一个 user 消息 (并行工具调用)
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": record["tool_use_id"],
                    "content": record["content"],
                }
                # 将连续的 tool_result 合并到同一个 user 消息中
                # 判断条件: 上一条是 user, content 是列表,
                #           且第一个元素是 tool_result 类型
                if (messages and messages[-1]["role"] == "user"
                        and isinstance(messages[-1]["content"], list)
                        and messages[-1]["content"]
                        and isinstance(messages[-1]["content"][0], dict)
                        and messages[-1]["content"][0].get("type") == "tool_result"):
                    # 追加到现有的 user 消息
                    messages[-1]["content"].append(result_block)
                else:
                    # 创建新的 user 消息
                    messages.append({
                        "role": "user",
                        "content": [result_block],
                    })

        return messages

    def list_sessions(self) -> list[tuple[str, dict]]:
        """列出所有 session, 按最近活跃时间排序。"""
        items = list(self._index.items())
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items


def _serialize_messages_for_summary(messages: list[dict]) -> str:
    """
    将消息列表扁平化为纯文本, 用于 LLM 摘要。

    把结构化的 API 消息格式转为人类可读的对话文本,
    这样摘要用的 LLM 就能理解对话内容并生成摘要。

    示例输出:
        [user]: 帮我看看 main.py
        [assistant]: 让我看看那个文件...
        [assistant called read_file]: {"file_path": "main.py"}
        [tool_result]: import os\nimport sys\n...
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            # 纯文本消息
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            # 内容块列表
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block['text']}")
                    elif btype == "tool_use":
                        parts.append(
                            f"[{role} called {block.get('name', '?')}]: "
                            f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                        )
                    elif btype == "tool_result":
                        # 工具结果可能很长, 只预览前 500 字符
                        rc = block.get("content", "")
                        preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                        parts.append(f"[tool_result]: {preview}")
                elif hasattr(block, "text"):
                    # SDK 对象 (非 dict), 用属性访问
                    parts.append(f"[{role}]: {block.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ContextGuard -- 上下文溢出保护
# ---------------------------------------------------------------------------
# 问题: LLM 有上下文窗口限制 (Claude ~200K tokens)
#       如果对话太长, API 调用会因为 token 超限而报错
#
# 解决方案: 三阶段重试机制
#   阶段 1: 正常调用 (大多数情况到这里就够了)
#   阶段 2: 如果 token 超限, 截断过大的工具结果
#           (有些工具返回超大输出, 截断后可能就放得下了)
#   阶段 3: 如果还超限, 用 LLM 把旧对话压缩成摘要
#           (牺牲早期细节, 保留最近对话)
#   阶段 4: 仍然超限, 抛异常 (实在没办法了)
#
# ┌──────────────────────────────────────────────────────────┐
# │  API 调用 → token 超限?                                  │
# │                 │                                        │
# │                Yes → 阶段1: 截断大 tool_result → 重试     │
# │                       │                                  │
# │                      仍超限 → 阶段2: LLM 摘要压缩 → 重试  │
# │                                │                         │
# │                               仍超限 → 抛异常             │
# └──────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------


























