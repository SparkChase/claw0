"""
第03节: 会话与上下文保护
"会话是 JSONL 文件。写入时追加, 读取时重放。过大时进行摘要压缩。"

围绕同一 agent 循环的两层机制:

  SessionStore -- JSONL 持久化 (写入时追加, 读取时重放)
  ContextGuard -- 三阶段溢出重试:
    先正常调用 -> 截断工具结果 -> 压缩历史 (50%) -> 失败

    用户输入
        |
    load_session() --> 从 JSONL 重建 messages[]
        |
    guard_api_call() --> 尝试 -> 截断 -> 压缩 -> 抛异常
        |
    save_turn() --> 追加到 JSONL
        |
    打印响应

用法:
    cd claw0
    python zh/s03_sessions.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

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

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

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
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

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


class ContextGuard:
    """保护 agent 免受上下文窗口溢出。"""

    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        """
        参数:
            max_tokens: 上下文窗口的安全上限
                        低于模型实际上限, 留余量给系统提示词和工具定义
        """
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        粗略估算文本的 token 数。

        经验值: 1 个 token ≈ 4 个英文字符 ≈ 0.75 个英文单词
        这只是估算, 实际需要 tokenizer 才能精确计算。
        但对判断 "是否接近上限" 已经够用了。
        """
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """
        估算整个消息列表的 token 数。

        遍历每条消息, 对所有文本内容进行估算:
          - 纯文本消息: 估算 content 字符串
          - 内容块列表: 估算每个 text/tool_result/tool_use 块
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {}))
                            )
                    else:
                        # SDK 对象 (Anthropic 返回的 Block 对象)
                        if hasattr(block, "text"):
                            total += self.estimate_tokens(block.text)
                        elif hasattr(block, "input"):
                            total += self.estimate_tokens(
                                json.dumps(block.input)
                            )
        return total

    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        """
        在换行边界处只保留头部进行截断。

        为什么在换行处截断?
          - 在换行处截断不会把一行内容劈成两半
          - 对于代码/日志等结构化文本, 保持行完整性很重要

        max_fraction: 工具结果最多占上下文的多少比例 (默认 30%)
        超过此比例的结果会被截断
        """
        max_chars = int(self.max_tokens * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        # 找到不超过 max_chars 的最后一个换行符, 在那里截断
        cut = result.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        head = result[:cut]
        return head + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"

    def compact_history(self, messages: list[dict],
                        api_client: Anthropic, model: str) -> list[dict]:
        """
        将前 50% 的消息压缩为 LLM 生成的摘要。
        保留最后 N 条消息 (N = max(4, 总数的 20%)) 不变。

        ★ 这是上下文管理的核心策略 ★

        为什么压缩前 50%?
          - 早期对话通常是背景信息, 不需要逐字保留
          - 最近对话包含当前任务的关键上下文, 不能动
          - 50% 是经验值, 平衡了信息保留和压缩效果

        压缩结果:
          原始: [msg1, msg2, ..., msg10]
          压缩后: [user: "摘要...", assistant: "明白了"], [msg7, msg8, msg9, msg10]
                  ↑ 摘要替代了前 6 条消息                              ↑ 保留最近 4 条

        参数:
            messages:    当前消息列表
            api_client:  Anthropic 客户端 (用于调用 LLM 生成摘要)
            model:       使用的模型
        """
        total = len(messages)
        if total <= 4:
            return messages

        # 保留最近的消息数: 至少 4 条, 或总数的 20%
        keep_count = max(4, int(total * 0.2))
        # 要压缩的消息数: 前面的 50%
        compress_count = max(2, int(total * 0.5))
        # 不能压缩太多, 至少保留 keep_count 条
        compress_count = min(compress_count, total - keep_count)

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]

        # 把旧消息序列化为文本, 让 LLM 生成摘要
        old_text = _serialize_messages_for_summary(old_messages)

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            # 调用 LLM 生成摘要 (这是一次额外的 API 调用, 消耗 token)
            summary_resp = api_client.messages.create(
                model=model,
                max_tokens=2048,
                system="You are a conversation summarizer. Be concise and factual.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_text = ""
            for block in summary_resp.content:
                if hasattr(block, "text"):
                    summary_text += block.text

            print_session(
                f"  [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            # 摘要生成失败: 直接丢弃旧消息 (总比崩溃好)
            print_warn(f"  [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        # 构造压缩后的消息列表:
        # 1. 一条 user 消息: 包含摘要内容
        # 2. 一条 assistant 消息: 模型确认理解了摘要
        #    (确保 user/assistant 交替, 满足 API 格式要求)
        # 3. 保留的最近消息
        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood, I have the context from our previous conversation."}],
            },
        ]
        compacted.extend(recent_messages)
        return compacted

    def _truncate_large_tool_results(self, messages: list[dict]) -> list[dict]:
        """
        遍历消息列表, 截断过大的 tool_result 块。

        这是溢出保护的第一道防线:
          - 工具输出往往是 token 大头 (比如读了一个大文件)
          - 截断它们通常就能解决超限问题
          - 不需要压缩整个对话历史
        """
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("content"), str)):
                        # 复制一份再修改, 不改变原始消息
                        block = dict(block)
                        block["content"] = self.truncate_tool_result(
                            block["content"]
                        )
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    def guard_api_call(
        self,
        api_client: Anthropic,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 2,
    ) -> Any:
        """
        ★ 带溢出保护的 API 调用 ★

        三阶段重试:
          第0次尝试: 正常调用 (大多数情况到这里就成功了)
          第1次尝试: 截断过大的工具结果 (针对工具输出过大的情况)
          第2次尝试: 通过 LLM 摘要压缩历史 (对话本身太长)

        参数:
            api_client:  Anthropic 客户端
            model:       模型 ID
            system:      系统提示词
            messages:    消息列表 (会被原地修改)
            tools:       工具定义列表
            max_retries: 最大重试次数 (默认 2, 即总共 3 次尝试)

        返回:
            API 响应对象

        异常:
            如果所有重试都失败, 抛出最后一次的异常
        """
        current_messages = messages

        for attempt in range(max_retries + 1):
            try:
                # 构建 API 调用参数
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 8096,
                    "system": system,
                    "messages": current_messages,
                }
                if tools:
                    kwargs["tools"] = tools
                result = api_client.messages.create(**kwargs)

                # ★ 成功! 如果中间修改过 messages, 同步回原始列表 ★
                # current_messages 可能是截断/压缩后的新列表
                # 需要把修改同步回外层的 messages 引用
                if current_messages is not messages:
                    messages.clear()
                    messages.extend(current_messages)
                return result

            except Exception as exc:
                # 判断是否是 token 超限错误
                # Anthropic API 的超限错误通常包含 "context" 或 "token" 关键词
                error_str = str(exc).lower()
                is_overflow = ("context" in error_str or "token" in error_str)

                if not is_overflow or attempt >= max_retries:
                    # 不是超限错误, 或重试用完了, 直接抛出
                    raise

                if attempt == 0:
                    # 第一次重试: 截断大工具结果
                    print_warn(
                        "  [guard] Context overflow detected, "
                        "truncating large tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(
                        current_messages
                    )
                elif attempt == 1:
                    # 第二次重试: LLM 摘要压缩
                    print_warn(
                        "  [guard] Still overflowing, "
                        "compacting conversation history..."
                    )
                    current_messages = self.compact_history(
                        current_messages, api_client, model
                    )

        raise RuntimeError("guard_api_call: exhausted retries")


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
# s03 的工具集比 s02 简化了一些 (去掉了 bash 和 edit_file)
# 因为这一节的重点是会话管理和上下文保护, 不是工具本身
# ---------------------------------------------------------------------------


def tool_read_file(file_path: str) -> str:
    """读取文件内容."""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_list_directory(directory: str = ".") -> str:
    """列出目录内容, 区分文件和子目录."""
    print_tool("list_directory", directory)
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            prefix = "[dir]  " if entry.is_dir() else "[file] "
            lines.append(prefix + entry.name)
        return "\n".join(lines) if lines else "[empty directory]"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_get_current_time() -> str:
    """获取当前 UTC 时间 (无参数工具的示例)."""
    print_tool("get_current_time", "")
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# 工具 schema + 分发表
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file under the workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to workspace directory.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory under workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path relative to workspace directory. Default is root.",
                },
            },
            # required 为空列表: directory 参数可选 (有默认值 ".")
            "required": [],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in UTC.",
        "input_schema": {
            "type": "object",
            # 无参数工具: properties 为空字典, required 为空列表
            "properties": {},
            "required": [],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": tool_read_file,
    "list_directory": tool_list_directory,
    "get_current_time": tool_get_current_time,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """根据工具名分发到对应的处理函数 (同 s02)."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# REPL 命令
# ---------------------------------------------------------------------------
# s03 新增的功能: 用 / 前缀的命令管理会话
# 这些命令不发送给 LLM, 而是直接在本地处理
#
# 命令列表:
#   /new [label]       创建新会话
#   /list              列出所有会话
#   /switch <id>       切换到某个会话 (支持前缀匹配)
#   /context           查看上下文 token 使用量
#   /compact           手动压缩对话历史
#   /help              显示帮助
# ---------------------------------------------------------------------------

def handle_repl_command(
    command: str,
    store: SessionStore,
    guard: ContextGuard,
    messages: list[dict],
) -> tuple[bool, list[dict]]:
    """
    处理以 / 开头的命令。

    参数:
        command:  用户输入的完整命令 (如 "/new 写代码")
        store:    SessionStore 实例
        guard:    ContextGuard 实例
        messages: 当前消息列表

    返回:
        (是否已处理, 更新后的 messages)
        如果已处理, 外层循环会 continue, 不发送给 LLM
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"  Created new session: {sid}" + (f" ({label})" if label else ""))
        # 新会话, 清空消息列表
        return True, []

    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("  No sessions found.")
            return True, messages

        print_info("  Sessions:")
        for sid, meta in sessions:
            active = " <-- current" if sid == store.current_session_id else ""
            label = meta.get("label", "")
            label_str = f" ({label})" if label else ""
            count = meta.get("message_count", 0)
            last = meta.get("last_active", "?")[:19]
            print_info(
                f"    {sid}{label_str}  "
                f"msgs={count}  last={last}{active}"
            )
        return True, messages

    elif cmd == "/switch":
        if not arg:
            print_warn("  Usage: /switch <session_id>")
            return True, messages
        # 支持前缀匹配: 输入 "abc" 可以匹配 "abc123def"
        # 这样不需要输入完整的 session ID
        target_id = arg.strip()
        matched = [
            sid for sid in store._index if sid.startswith(target_id)
        ]
        if len(matched) == 0:
            print_warn(f"  Session not found: {target_id}")
            return True, messages
        if len(matched) > 1:
            # 前缀不唯一, 提示所有匹配项
            print_warn(f"  Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages

        sid = matched[0]
        # 从 JSONL 重建消息列表
        new_messages = store.load_session(sid)
        print_session(f"  Switched to session: {sid} ({len(new_messages)} messages)")
        return True, new_messages

    elif cmd == "/context":
        # 显示上下文使用量 (带进度条)
        estimated = guard.estimate_messages_tokens(messages)
        pct = (estimated / guard.max_tokens) * 100
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        # 颜色: <50% 绿色, 50-80% 黄色, >80% 红色
        color = GREEN if pct < 50 else (YELLOW if pct < 80 else RED)
        print_info(f"  Context usage: ~{estimated:,} / {guard.max_tokens:,} tokens")
        print(f"  {color}[{bar}] {pct:.1f}%{RESET}")
        print_info(f"  Messages: {len(messages)}")
        return True, messages

    elif cmd == "/compact":
        if len(messages) <= 4:
            print_info("  Too few messages to compact (need > 4).")
            return True, messages
        print_session("  Compacting history...")
        new_messages = guard.compact_history(messages, client, MODEL_ID)
        print_session(f"  {len(messages)} -> {len(new_messages)} messages")
        return True, new_messages

    elif cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages

    return False, messages


# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------
# 与 s01/s02 相同的 while True 循环, 加入了 SessionStore + ContextGuard。
#
# 与 s02 相比的变化:
#   1. 启动时恢复最近的会话 (不再从空对话开始)
#   2. 每轮对话都保存到 JSONL (持久化)
#   3. API 调用通过 guard_api_call 保护 (自动处理溢出)
#   4. 支持 / 命令管理会话
#
# ┌─────────────────────────────────────────────────────────────┐
# │  消息在三个地方流动:                                         │
# │                                                             │
# │  1. 内存中的 messages[]   ← API 直接使用的格式              │
# │  2. JSONL 文件            ← 持久化存储 (追加写入, 按需重建)  │
# │  3. API 响应              ← LLM 返回的对象                  │
# │                                                             │
# │  写入: 内存 → JSONL (save_turn / save_tool_result)          │
# │  读取: JSONL → 内存 (_rebuild_history)                      │
# │  调用: 内存 → API (guard_api_call)                          │
# └─────────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """带会话持久化和上下文保护的主 agent 循环。"""

    # 初始化 session 存储和上下文保护
    store = SessionStore(agent_id="claw0")
    guard = ContextGuard()

    # 恢复最近的会话或创建新会话
    # 这样重启程序后能接着上次的对话继续
    sessions = store.list_sessions()
    if sessions:
        # 有历史 session, 恢复最近活跃的那个
        sid = sessions[0][0]
        messages = store.load_session(sid)
        print_session(f"  Resumed session: {sid} ({len(messages)} messages)")
    else:
        # 没有 session, 创建一个初始的
        sid = store.create_session("initial")
        messages = []
        print_session(f"  Created initial session: {sid}")

    print_info("=" * 60)
    print_info("  claw0  |  Section 03: Sessions & Context Guard")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Session: {store.current_session_id}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  Type /help for commands, quit/exit to leave.")
    print_info("=" * 60)
    print()

    # ================================================================
    # 外层循环: 等待用户输入
    # ================================================================
    while True:
        # --- 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        # --- REPL 命令处理 (s03 新增) ---
        if user_input.startswith("/"):
            handled, messages = handle_repl_command(
                user_input, store, guard, messages
            )
            if handled:
                continue

        # --- 追加用户消息 ---
        # 和 s02 一样追加到 messages, 但现在同时保存到 JSONL
        messages.append({
            "role": "user",
            "content": user_input,
        })
        store.save_turn("user", user_input)

        # ================================================================
        # 内层循环: 工具调用链
        # ================================================================
        while True:
            try:
                # ★ 用 guard_api_call 替代了直接的 client.messages.create ★
                # 这样 token 超限时自动截断/压缩, 不会直接报错
                response = guard.guard_api_call(
                    api_client=client,
                    model=MODEL_ID,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            # ★ 保存 assistant 回复到 JSONL ★
            # API 返回的是 SDK 对象 (TextBlock, ToolUseBlock),
            # 需要序列化为 dict 才能存入 JSONL
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # 将内容块序列化为 JSONL 存储格式
            # 因为 SDK 对象不能直接 json.dumps, 需要手动转为 dict
            serialized_content = []
            for block in response.content:
                if hasattr(block, "text"):
                    serialized_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    serialized_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            store.save_turn("assistant", serialized_content)

            # --- 检查 stop_reason ---
            if response.stop_reason == "end_turn":
                # 模型回复完成, 提取文本打印
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                # 跳出内循环, 回到外层等待用户输入
                break

            elif response.stop_reason == "tool_use":
                # 模型请求调用工具
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    # 执行工具
                    result = process_tool_call(block.name, block.input)

                    # ★ 保存工具调用和结果到 JSONL ★
                    # 写入两行: tool_use 请求 + tool_result 结果
                    store.save_tool_result(
                        block.id, block.name, block.input, result
                    )

                    # 构造 API 格式的 tool_result 块
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                # 把所有工具结果作为 user 消息追加
                # (Anthropic API 要求 tool_result 在 user 角色中)
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                # 继续内循环 -- 模型会看到工具结果并决定下一步
                continue

            else:
                # max_tokens 或其他情况
                print_info(f"[stop_reason={response.stop_reason}]")
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                break


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """程序入口: 检查 API key, 启动 agent 循环."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


# Python 的标准入口模式:
# 只有直接运行此文件时才会执行 main()
# 如果被其他文件 import, 不会自动运行
if __name__ == "__main__":
    main()
