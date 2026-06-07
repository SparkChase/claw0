# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import os
import sys
import subprocess
from pathlib import Path
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
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

# 工具输出最大字符数 -- 防止超大输出撑爆上下文
# LLM 有 token 限制, 如果工具返回了几 MB 的输出塞进 messages,
# 后续 API 调用会因为 token 超限而失败
MAX_TOOL_OUTPUT = 50000

# 工作目录 -- 所有文件操作相对于此目录, 防止路径穿越
# 用 Path.cwd() 获取当前工作目录 (运行脚本时所在的目录)
WORKDIR = Path.cwd()

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"       # 红色 -- s01 没用到, s02 用于错误高亮
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    """打印工具调用信息. 用暗淡样式, 显示正在执行什么工具.

    效果:
      [tool: bash] ls -la
      [tool: read_file] src/main.py
    """
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")



# ---------------------------------------------------------------------------
# 安全辅助函数
# ---------------------------------------------------------------------------
# 这两个函数是工具的 "安全护栏":
#   - safe_path:  防止模型读写工作目录之外的文件 (路径穿越攻击)
#   - truncate:   防止工具输出过大撑爆上下文
#
# 为什么需要这些? 因为模型生成的工具参数可能不安全!
# 比如用户可能诱导模型读取 /etc/passwd 或执行危险命令.
# 这些是最低限度的防护, 生产环境需要更多安全措施.
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """截断过长的输出, 并附上提示.

    为什么需要截断:
        - 工具可能返回超大输出 (比如 cat 一个大文件)
        - LLM 有 token 限制, 超大输出会导致 API 调用失败
        - 截断后附上总字符数, 让模型知道输出被截了
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------
# 每个工具函数接收关键字参数 (和 schema 中的 properties 对应),
# 返回字符串结果. 错误通过返回 "Error: ..." 传递给模型.
#
# 设计原则:
#   - 工具不抛异常给调用者, 而是把错误信息作为字符串返回
#   - 这样模型能 "看到" 错误并自行修正 (比如换一个路径重试)
#   - 如果抛异常, 整个 agent 循环可能崩溃, 模型也没机会纠正
# ---------------------------------------------------------------------------

def tool_bash(command: str, timeout: int = 30) -> str:
    """执行 shell 命令并返回输出.

    参数:
        command: 要执行的 shell 命令
        timeout: 超时秒数, 默认 30 秒

    返回:
        命令的 stdout + stderr 输出, 或者 "Error: ..." 错误信息

    安全措施:
        1. 危险命令黑名单检查 (最基础的防护)
        2. 超时限制 (防止命令挂住)
        3. 在 WORKDIR 下执行 (限制工作目录)
    """
    # 基础安全检查: 拒绝明显危险的命令
    # 这只是最基本的黑名单, 生产环境需要更严格的安全措施
    # (比如用容器沙箱、白名单等)
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

    print_tool("bash", command)
    try:
        # subprocess.run: 执行外部命令
        # 参数:
        #   shell=True:     通过 shell 执行 (支持管道、通配符等)
        #   capture_output=True: 捕获 stdout 和 stderr
        #   text=True:      以文本模式返回 (而非 bytes)
        #   timeout:        超时秒数
        #   cwd:            工作目录
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKDIR),
        )
        # 拼接输出: stdout + stderr
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        # 非零退出码表示命令失败, 附上退出码让模型知道
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"



def tool_read_file(file_path: str) -> str:
    """读取文件内容.

    参数:
        file_path: 文件路径 (相对于 WORKDIR)

    返回:
        文件的文本内容, 或者 "Error: ..." 错误信息

    安全措施:
        - safe_path() 确保不会读取工作目录之外的文件
    """
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        # safe_path 抛出的路径穿越错误
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(file_path: str, content: str) -> str:
    """写入内容到文件. 父目录不存在时自动创建.

    参数:
        file_path: 文件路径 (相对于 WORKDIR)
        content:   要写入的内容

    返回:
        成功/失败信息

    注意: 会覆盖已有文件内容!
    """
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path)
        # mkdir(parents=True): 自动创建所有不存在的父目录
        # exist_ok=True: 如果目录已存在不报错
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"

def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """
    精确替换文件中的文本.
    old_string 必须在文件中恰好出现一次, 否则报错.
    这和 OpenClaw 的 edit 工具逻辑一致.

    参数:
        file_path:   文件路径 (相对于 WORKDIR)
        old_string:  要查找的原始文本 (必须精确匹配, 包括空格/缩进)
        new_string:  替换后的新文本

    为什么要求 old_string 唯一:
        - 如果出现多次, 替换哪一个? 模型可能猜错, 导致文件被改坏
        - 强制唯一性, 确保替换是确定性的
        - 如果不唯一, 模型需要提供更多上下文来定位

    这就是为什么系统提示词里说 "Always read a file before editing it":
        - 模型必须先读取文件, 拿到精确的文本
        - 然后才能构造出精确匹配的 old_string
        - 如果凭记忆编写 old_string, 大概率会不匹配
    """
    print_tool("edit_file", f"{file_path} (replace {len(old_string)} chars)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"

        content = target.read_text(encoding="utf-8")
        # 统计 old_string 在文件中出现的次数
        count = content.count(old_string)

        if count == 0:
            # 没找到 -> old_string 和文件内容不匹配
            # 常见原因: 模型没先读文件, 凭记忆写了 old_string
            return "Error: old_string not found in file. Make sure it matches exactly."
        if count > 1:
            # 找到多次 -> old_string 不够唯一
            # 模型需要提供更多上下文来精确定位
            return (
                f"Error: old_string found {count} times. "
                "It must be unique. Provide more surrounding context."
            )

        # 精确替换: 只替换第一个匹配 (因为已经确认只有一个)
        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"



# ---------------------------------------------------------------------------
# 工具定义: Schema (传给 API) + Handler 调度表
# ---------------------------------------------------------------------------
# 关键认知:
#   TOOLS 数组 = 告诉模型 "你有哪些工具可用"
#   TOOL_HANDLERS 字典 = 告诉我们的代码 "收到工具调用时执行什么函数"
#   两者通过 name 字段关联. 就这么简单.
#
# ┌───────────────────────────────────────────────────────────────┐
# │  TOOLS (schema)          TOOL_HANDLERS (实现)                │
# │                                                               │
# │  { name: "bash",          "bash" -> tool_bash()              │
# │    input_schema: {...} }  "read_file" -> tool_read_file()   │
# │                          "write_file" -> tool_write_file()  │
# │  { name: "read_file",    "edit_file" -> tool_edit_file()   │
# │    input_schema: {...} }                                      │
# │                                                               │
# │  ↳ 传给 API, 让模型知道    ↳ 代码收到工具调用时查表执行        │
# │    可用工具和参数格式         实际的工具函数                    │
# └───────────────────────────────────────────────────────────────┘
#
# input_schema 遵循 JSON Schema 规范:
#   type: # object          -> 参数是一个对象 (即关键字参数)
#   properties: {...}        -> 每个参数的名字、类型、描述
#   required: [...]          -> 哪些参数是必填的
#
# 模型会根据 schema 生成符合格式的 tool_input,
# 我们的代码用 handler(**tool_input) 把它展开为关键字参数调用.
# ---------------------------------------------------------------------------


TOOLS = [
    {
        # 工具名: 必须和 TOOL_HANDLERS 的 key 一致
        "name": "bash",
        # 工具描述: 告诉模型这个工具做什么, 什么时候该用
        # 描述越清晰, 模型越能正确选择工具
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        # 输入参数的 JSON Schema
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],  # command 必填, timeout 可选 (有默认值)
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["file_path", "content"],  # 两个参数都必填
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with a new string. "
            "The old_string must appear exactly once in the file. "
            "Always read the file first to get the exact text to replace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],  # 三个都必填
        },
    },
]


# 调度表: 工具名 -> 处理函数
# 当模型返回 tool_use block 时, 我们用 block.name 在这里查到对应的函数
# 然后调用 handler(**block.input) 执行
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}


# ---------------------------------------------------------------------------
# 工具调用处理
# ---------------------------------------------------------------------------


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    根据工具名分发到对应的处理函数.
    这就是整个 "agent" 的核心调度逻辑 -- 一个查表调用.

    参数:
        tool_name:  模型请求调用的工具名 (如 "bash", "read_file")
        tool_input: 模型生成的工具参数 (dict, 如 {"command": "ls -la"})

    返回:
        工具执行结果字符串

    工作流程:
        1. 在 TOOL_HANDLERS 中查找工具名 -> 得到处理函数
        2. 如果找不到, 返回错误 (模型看到后会自行修正)
        3. 用 **tool_input 把字典展开为关键字参数调用函数
           例: tool_input={"command": "ls", "timeout": 10}
               等价于 tool_bash(command="ls", timeout=10)
        4. TypeError: 参数不匹配 (比如缺少必填参数)
        5. 其他异常: 工具执行出错
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        # 未知工具 -- 模型可能产生了幻觉, 请求了一个不存在的工具
        # 返回错误让它自行纠正
        return f"Error: Unknown tool '{tool_name}'"
    try:
        # **tool_input: 把字典展开为关键字参数
        # 这是 schema 和实现之间的桥梁:
        #   schema 定义了参数名和类型 -> 模型生成符合格式的 dict
        #   **展开 -> 传给函数 -> 函数按参数名接收
        return handler(**tool_input)
    except TypeError as exc:
        # 参数不匹配: 缺少必填参数, 或传了函数不接受的参数
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"



# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------
# 和 s01 的区别:
#   1. API 调用时传入 tools=TOOLS  (告诉模型有哪些工具)
#   2. stop_reason == "tool_use" 时, 执行工具并把结果送回模型
#   3. 用一个内层 while 循环处理连续工具调用 (模型可能连续调多个工具)
#
# 循环结构本身没变. 这就是 agent 的本质:
#   一个 while 循环 + 一张调度表.
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  双层循环结构:                                                   │
# │                                                                 │
# │  外层 while True:          ← 等待用户输入                       │
# │      获取用户输入                                               │
# │      追加 user 消息                                             │
# │                                                                 │
# │      内层 while True:      ← 处理工具调用链                     │
# │          调用 LLM API                                          │
# │          if end_turn:  打印回复, break 内层                     │
# │          if tool_use:  执行工具, 追加结果, continue 内层         │
# │                                                                 │
# │  为什么需要内层循环?                                             │
# │  因为模型可能需要连续调用多个工具才能完成任务:                     │
# │    用户: "帮我修复 bug"                                         │
# │    → LLM: read_file(main.py)        [tool_use]                 │
# │    → LLM: read_file(utils.py)       [tool_use]                 │
# │    → LLM: edit_file(main.py, ...)   [tool_use]                 │
# │    → LLM: "已经修复了!"             [end_turn]                  │
# └─────────────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    """主 agent 循环 -- 带工具的 REPL."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 02: 工具使用")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workdir: {WORKDIR}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  输入 'quit' 或 'exit' 退出, Ctrl+C 同样有效.")
    print_info("=" * 60)
    print()

    # ================================================================
    # 外层循环: 等待用户输入
    # ================================================================
    while True:
        # --- Step 1: 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}再见.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- Step 2: 追加 user 消息 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

# ================================================================
        # 内层循环: 处理 LLM 回复 + 工具调用链
        # ================================================================
        # 模型可能连续调用多个工具才最终给出文本回复.
        # 所以我们用 while True 循环, 直到 stop_reason != "tool_use"
        while True:
            try:
                # ★ 和 s01 的关键区别: 传入了 tools=TOOLS ★
                # 这让模型知道它有哪些工具可用
                # 当模型决定使用工具时, stop_reason 会变成 "tool_use"
                # 并且 response.content 中会包含 ToolUseBlock
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=8096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,       # ← 新增! 告诉模型可用的工具列表
                    messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                # 出错时回滚本轮所有消息到最近的 user 消息
                # 为什么比 s01 复杂? 因为内层循环可能已经追加了
                # assistant 消息和 tool_result 消息, 需要全部撤回
                while messages and messages[-1]["role"] != "user":
                    messages.pop()  # 弹出 assistant / tool_result 消息
                if messages:
                    messages.pop()  # 弹出触发这轮的 user 消息
                break

            # ★ 无论 stop_reason 是什么, 都必须把 assistant 回复存入历史 ★
            # 因为后续的 tool_result 消息必须紧跟在对应的 assistant 消息之后
            # Anthropic API 的消息顺序要求:
            #   user -> assistant (含 tool_use) -> user (含 tool_result) -> ...
            # 如果不存 assistant 消息, 下一轮的 tool_result 就没地方挂
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # --- 检查 stop_reason ---
            if response.stop_reason == "end_turn":
                # 模型说完了, 提取文本打印
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                # 跳出内循环, 回到外层等待下一次用户输入
                break

            elif response.stop_reason == "tool_use":
                # 模型想调用工具!
                # response.content 可能包含多个 tool_use block (并行调用)
                # 比如 LLM 可能一次请求读取多个文件:
                #   [TextBlock("让我看看这些文件..."), ToolUseBlock("read_file", ...), ToolUseBlock("read_file", ...)]
                tool_results = []

                for block in response.content:
                    # 跳过非 tool_use 的块 (比如模型可能先说一句话再调工具)
                    if block.type != "tool_use":
                        continue

                    # 执行工具
                    # block.name:  工具名 (如 "bash", "read_file")
                    # block.input: 工具参数 (dict, 如 {"command": "ls"})
                    # block.id:    这次工具调用的唯一标识, 用于关联结果
                    result = process_tool_call(block.name, block.input)

                    # ★ 构造 tool_result 消息块 ★
                    # 必须包含 tool_use_id, 把结果和请求对应起来
                    # 格式要求:
                    #   type: "tool_result"     -- 表示这是工具执行结果
                    #   tool_use_id: block.id   -- 对应哪次工具调用
                    #   content: result         -- 工具返回的内容
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                # ★ 把所有工具结果作为一条 user 消息追加 ★
                # Anthropic API 的硬性要求:
                #   tool_result 必须放在 role="user" 的消息中
                #   而不是单独的 role="tool"  (那是 OpenAI 的做法)
                #
                # 消息流:
                #   user: "帮我读文件"
                #   assistant: [ToolUseBlock("read_file", {file_path: "a.py"})]
                #   user: [tool_result {tool_use_id: "...", content: "文件内容..."}]
                #     ↑ 工具结果伪装成 user 消息!
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                # 继续内循环 -- 模型会看到工具结果并决定下一步
                # 可能:
                #   - 再调一个工具 (继续 tool_use)
                #   - 给出最终回复 (end_turn)
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
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY 未设置.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的 key.{RESET}")
        sys.exit(1)

    agent_loop()


# Python 的标准入口模式:
# 只有直接运行此文件时才会执行 main()
# 如果被其他文件 import, 不会自动运行
if __name__ == "__main__":
    main()
