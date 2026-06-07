# yjy i love u
#==========
# 导入
#==========
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from anthropic import  Anthropic


load_dotenv(Path(__file__).resolve().parent.parent / ".env",override=True)

MODEL_ID = os.getenv("MODEL_ID")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = "你是trace,你很可爱，永远积极向上，能够跟用户聊很深度的内容，有自己的想法和见解给人启迪"


# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
# 终端通过 ANSI 转义序列来控制文字样式
# 格式: \033[<代码>m   其中 \033 是 ESC 字符
# 多个代码用分号连接, 比如 \033[1;36m = 加粗+青色
# RESET (\033[0m) 用于恢复默认样式, 防止后续文字也被染色

CYAN = "\033[36m"     # 青色 -- 用于用户输入提示符
GREEN = "\033[32m"    # 绿色 -- 用于助手回复
YELLOW = "\033[33m"   # 黄色 -- 用于警告/错误
DIM = "\033[2m"       # 暗淡 -- 用于辅助信息 (版权、提示等)
RESET = "\033[0m"     # 重置所有样式
BOLD = "\033[1m"      # 加粗

def colored_prompt()-> str:
    """
    返回带颜色的用户输入提示符：
    效果：用青色加粗显示 “yjy say > ”，然后恢复默认样式等待输入
    """
    return f"{CYAN}{BOLD}yjy say > {RESET}"

def print_assistant(text: str) -> None:
    """
    打印助手回复：用绿色加粗标题+正文：
    效果：
    trace's Assistant： 你好 ！
    """
    print(f"\n{GREEN}{BOLD}trace's Assistant:{RESET} {text}\n")

def print_info(text: str) -> None:
    """打印辅助信息, 用暗淡样式 (不太醒目, 适合提示文字)."""
    print(f"{DIM}{text}{RESET}")





# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------
#   1. 收集用户输入, 追加到 messages
#   2. 调用 API
#   3. 检查 stop_reason 决定下一步
#
#   本节 stop_reason 永远是 "end_turn" (没有工具).
#   下一节加入 "tool_use" -- 循环结构保持不变.
#
# ┌─────────────────────────────────────────────────────────┐
# │  Agent 循环的本质就是:                                    │
# │                                                         │
# │  while True:                                            │
# │      用户输入 -> 放进 messages                            │
# │      把 messages 发给 LLM                               │
# │      看 LLM 返回的 stop_reason:                         │
# │          "end_turn"  -> 展示回复, 等下一轮输入            │
# │          "tool_use"  -> 执行工具, 把结果放回 messages,    │
# │                          再调一次 LLM (下一节)           │
# │                                                         │
# │  所有 Agent 框架 (LangChain, CrewAI, AutoGPT...)        │
# │  底层都是这个循环, 只是包装了更多功能.                     │
# └─────────────────────────────────────────────────────────┘
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    """
    主agent循环--对话式 REPL
    read eval print loop
    """
    messages: list[dict]=[]

    print_info("="*60)
    print_info("  yjy  |  Section 01: Agent 循环")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  输入 'quit' 或 'exit' 退出. Ctrl+C 同样有效.")
    print_info("="*60)

    while True:
        # --- 获取用户输入 ---
        # input() 会阻塞等待用户输入, 并返回输入的字符串
        # colored_prompt() 提供一个彩色的提示符
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            # KeyboardInterrupt: 用户按了 Ctrl+C
            # EOFError: 输入流结束 (比如管道输入用完了)
            print(f"\n{DIM}再见.{RESET}")
            break

        # 空输入就跳过, 不发送给 LLM
        if not user_input:
            continue

        # 输入 quit/exit 退出循环
        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- 追加到对话历史 ---
        # 把用户消息追加到 messages 列表
        # 格式: {"role": "user", "content": "用户说的内容"}
        # 这是 Anthropic API 要求的消息格式
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- 调用 LLM ---
        # client.messages.create() 是 Anthropic SDK 的核心方法
        # 参数:
        #   model:       使用哪个模型
        #   max_tokens:  最多生成多少 token (防止回复过长消耗太多额度)
        #   system:      系统提示词 (不在 messages 里, 单独传)
        #   messages:    完整对话历史
        #
        # 返回: 一个 Message 对象, 包含:
        #   .content:       内容块列表 (可能是文本块或工具调用块)
        #   .stop_reason:   停止原因 ("end_turn" / "tool_use" / "max_tokens" ...)
        #   .model:         实际使用的模型
        #   .usage:         token 用量统计
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            # API 调用可能失败: 网络错误、key 无效、超限等
            # 打印错误, 并把刚才追加的用户消息删掉 (因为没得到回复)
            # 这样下次循环用户可以重新输入, 不会带着失败的上下文
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()  # 撤回刚才 append 的用户消息
            continue

        # --- 检查 stop_reason ---
        # stop_reason 告诉我们 LLM 为什么停下来:
        #
        #   "end_turn"   -> 回复完成, 正常结束, 等待用户下一轮输入
        #   "tool_use"   -> LLM 想调用工具, 需要我们执行工具并返回结果
        #   "max_tokens" -> 达到 max_tokens 上限, 回复被截断
        #
        # 这是 Agent 的关键分支点:
        #   end_turn  -> 一轮对话结束, 回到用户输入
        #   tool_use  -> 需要继续循环 (执行工具 -> 再调 API -> 再判断)
        #               (本节暂不处理, 下一节实现)


        if response.stop_reason == "end_turn":
            # ---- 正常结束: 提取文本, 打印, 存入历史 ----
            # response.content 是一个内容块 (ContentBlock) 列表
            # 每个块可能是:
            #   TextBlock   -> 有 .text 属性, 包含文本内容
            #   ToolUseBlock -> 有 .id, .name, .input 属性 (下一节用)
            #
            # 这里遍历所有块, 把文本拼接起来
            # (本节没有工具, 所以只有 TextBlock)
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            # 打印助手回复
            print_assistant(assistant_text)

            # 把助手回复追加到对话历史
            # 注意: content 直接用 response.content (原始的块列表)
            # 而不是 assistant_text (纯字符串)
            # 因为 API 要求 assistant 消息的 content 格式与返回一致
            # 如果是纯文本, 传字符串也可以, 但用原始格式更通用
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        elif response.stop_reason == "tool_use":
            # ---- 工具调用: 本节不处理, 提示用户看下一节 ----
            # 虽然本节没有定义工具, 但 LLM 有时仍可能想调用内置能力
            # 把它的回复存入历史, 以保持对话连贯
            print_info("[stop_reason=tool_use] 本节没有可用工具.")
            print_info("参见 s02_tool_use.py 了解工具支持.")
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        else:
            # ---- 其他停止原因 (如 max_tokens) ----
            # 尽可能展示已有内容, 并存入历史
            print_info(f"[stop_reason={response.stop_reason}]")
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                print_assistant(assistant_text)
            messages.append({
                "role": "assistant",
                "content": response.content,
            })


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
# 这样这个文件的函数可以被其他模块复用
if __name__ == "__main__":
    main()
