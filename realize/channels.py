"""
第04节: 通道 -- "同一大脑, 多个嘴巴"

Channel 封装了平台差异, 使 agent 循环只看到统一的 InboundMessage。
添加新平台 = 实现 receive() + send(); 循环不需要改动。

这里的 "通道" 可以直接理解成一个社交媒体入口:
    - Telegram: 用户给 Telegram Bot 发消息, 本文件用 Bot API 拉取消息。
    - 飞书/Lark: 用户给飞书机器人发消息, 平台通过 webhook 推送事件。
    - CLI: 终端输入, 用来模拟一个最小平台。

每个平台原始字段都不一样:
    Telegram 用 update/message/chat/from/text。
    飞书用 event/message/sender/content。

Channel 的工作就是把这些平台字段翻译成统一的 InboundMessage:
    text       = 用户实际发来的文本
    sender_id  = 谁发的
    channel    = 来自哪个平台, 例如 "telegram" / "feishu"
    account_id = 哪个机器人账号收到的
    peer_id    = 应该把回复发回哪里, 私聊通常是用户 id, 群聊通常是群/会话 id

agent_loop 和 run_agent_turn 不需要知道 Telegram 或飞书 API 的细节。
它们只处理 InboundMessage, 最后再调用对应 Channel.send(peer_id, reply)
把回复送回原来的社交媒体会话。

    Telegram ----.                          .---- sendMessage API
    Feishu -------+-- InboundMessage ---+---- im/v1/messages
    CLI (stdin) --'    Agent Loop        '---- print(stdout)

运行方法:  cd claw0 && python zh/s04_channels.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
    # 可选: TELEGRAM_BOT_TOKEN, FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_ENCRYPT_KEY
"""

import json, os, sys, time, threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = WORKSPACE_DIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant connected to multiple messaging channels.\n"
    "You can save and search notes using the provided tools.\n"
    "When responding, be concise and helpful."
)

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
BOLD, RED, BLUE = "\033[1m", "\033[31m", "\033[34m"


def print_assistant(text: str, ch: str = "cli") -> None:
    prefix = f"[{ch}] " if ch != "cli" else ""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {prefix}{text}\n")

def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")

def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")

def print_channel(text: str) -> None:
    print(f"{BLUE}{text}{RESET}")

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class InboundMessage:
    """
    所有通道都规范化为此结构。Agent 循环只看到 InboundMessage。

    这是本节最关键的“平台适配层”:
    - Telegram 原始消息是 Bot API 的 update/message JSON。
    - 飞书原始消息是事件回调 payload。
    - CLI 原始消息只是 input() 得到的一行字符串。

    这些不同来源都会被整理成同一组字段。后面的 agent 代码只关心:
        inbound.text      用户说了什么
        inbound.peer_id   回复应该发回哪个会话
        inbound.channel   用哪个 Channel.send() 发回去
    """
    text: str
    # 平台里的“发送者”。Telegram 是 from.id; 飞书通常是 sender_id.open_id。
    sender_id: str
    # 平台名称。用它从 ChannelManager 取回对应的发送器, 例如 mgr.get("telegram")。
    channel: str = ""
    # 机器人账号。一个平台可以接多个 bot, account_id 用来区分是哪个 bot 收到消息。
    account_id: str = ""
    # 会话目标。回包时直接传给 send(to=peer_id)。私聊/群聊/话题会编码成不同格式。
    peer_id: str = ""
    # 是否来自群聊。群聊里常需要额外规则, 例如飞书只在 @bot 时才响应。
    is_group: bool = False
    # 附件的统一描述。教学版只保存 file_id/image_key 等引用, 不下载真实文件。
    media: list = field(default_factory=list)
    # 原始平台 payload。调试时能回头看平台到底发来了什么字段。
    raw: dict = field(default_factory=dict)

@dataclass
class ChannelAccount:
    """
    每个 bot 的配置。同一通道类型可以运行多个 bot。

    举例:
    - telegram/tg-primary 使用 TELEGRAM_BOT_TOKEN。
    - feishu/feishu-primary 使用 FEISHU_APP_ID + FEISHU_APP_SECRET。

    account_id 不是用户 id, 而是“我方机器人账号 id”。它用于区分多机器人场景,
    也会参与 session key, 避免不同机器人账号的上下文串在一起。
    """
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 会话键
# ---------------------------------------------------------------------------

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    # 会话键把“平台 + 机器人账号 + 对话对象”串起来。
    # 这样 Telegram 用户 A 和飞书用户 A 即使名字/数字 id 相同, 也不会共享上下文。
    #
    # 当前教学版固定 agent:main, 后面第05节会加入路由, 让不同 channel/peer
    # 可以被分配给不同 agent。
    return f"agent:main:direct:{channel}:{peer_id}"

# ---------------------------------------------------------------------------
# Channel 抽象基类
# ---------------------------------------------------------------------------

class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None:
        """
        从平台拿一条消息。

        - Telegram 是调用 getUpdates 长轮询。
        - CLI 是调用 input()。
        - 飞书 webhook 是外部 HTTP 服务收到事件后调用 parse_event(),
          所以本教学文件里的 receive() 返回 None。

        无论平台原始格式是什么, 返回值都必须是 InboundMessage。
        """
        ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        """
        把 agent 回复送回平台。

        - to 通常就是 inbound.peer_id。
        - Telegram 会把 to 转成 chat_id/message_thread_id, 再调用 sendMessage。
        - 飞书会把 to 转成 receive_id, 再调用 im/v1/messages。

        这就是代码和社交媒体平台真正发生连接的地方: Channel 内部调用平台 API。
        """
        ...

    def close(self) -> None:
        pass

# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------

class CLIChannel(Channel):
    name = "cli"

    def __init__(self) -> None:
        self.account_id = "cli-local"

    def receive(self) -> InboundMessage | None:
        # CLI 是最小的“通道”示例: 终端 input() 就相当于平台发来的消息。
        # 它没有 HTTP API, 但仍然遵守同一个 Channel 契约。
        try:
            text = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None
        return InboundMessage(
            # 这里手工填充字段, 对应 Telegram/飞书里由平台 payload 提供的字段。
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        # CLI 的“发送到平台”就是 print()。真实社交媒体会在 send() 中调用 HTTP API。
        print_assistant(text)
        return True

# ---------------------------------------------------------------------------
# 偏移量持久化 -- 两个简单函数
# ---------------------------------------------------------------------------

def save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(offset))

def load_offset(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# TelegramChannel -- Bot API 长轮询
# ---------------------------------------------------------------------------

class TelegramChannel(Channel):
    name = "telegram"
    MAX_MSG_LEN = 4096

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("TelegramChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        # Telegram Bot API 的所有请求都挂在这个 URL 下。
        # account.token 来自 TELEGRAM_BOT_TOKEN, 也就是 BotFather 给你的 bot token。
        # 有了 token, 这个程序才能代表该 Telegram bot 调 getUpdates/sendMessage。
        self.base_url = f"https://api.telegram.org/bot{account.token}"
        self._http = httpx.Client(timeout=35.0)
        raw = account.config.get("allowed_chats", "")
        # 可选白名单。生产环境通常不希望任何陌生群/用户都能唤起 bot。
        # TELEGRAM_ALLOWED_CHATS 里填 chat_id/user_id, 这里只处理这些会话。
        self.allowed_chats = {c.strip() for c in raw.split(",") if c.strip()} if raw else set()

        # getUpdates 使用 offset 表示“下一条要取的 update_id”。
        # 持久化到磁盘后, 程序重启不会重复消费旧消息。
        self._offset_path = STATE_DIR / "telegram" / f"offset-{self.account_id}.txt"
        self._offset = load_offset(self._offset_path)
        # _seen 是运行期去重。offset 负责跨重启, _seen 负责同一进程内防重复。
        self._seen: set[int] = set()
        # Telegram 相册/多图会拆成多条 message, 但它们共享 media_group_id。
        # 先短暂缓存, 再合并成一个 InboundMessage, agent 看到的就是一条用户输入。
        self._media_groups: dict[str, dict] = {}
        # 用户连续粘贴长文本时, Telegram 可能拆成多条消息。
        # 这里按 peer_id + sender_id 缓冲, 等 1s 没有新片段再一起交给 agent。
        self._text_buf: dict[tuple[str, str], dict] = {}

    def _api(self, method: str, **params: Any) -> dict:
        # TelegramChannel 和社交媒体平台的连接点之一:
        # 这里统一调用 Telegram Bot API, method 可能是 getUpdates/sendMessage/sendChatAction。
        filtered = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._http.post(f"{self.base_url}/{method}", json=filtered)
            data = resp.json()
            if not data.get("ok"):
                print(f"  {RED}[telegram] {method}: {data.get('description', '?')}{RESET}")
                return {}
            return data.get("result", {})
        except Exception as exc:
            print(f"  {RED}[telegram] {method}: {exc}{RESET}")
            return {}

    def send_typing(self, chat_id: str) -> None:
        # 给 Telegram 展示“bot 正在输入”的状态, 让用户知道请求已经被接收。
        self._api("sendChatAction", chat_id=chat_id, action="typing")

    def poll(self) -> list[InboundMessage]:
        # getUpdates 是 Telegram 的长轮询接口。
        # timeout=30 表示请求最多挂起 30 秒, 有新消息就立刻返回。
        # 返回的是 Telegram update JSON 列表, 还不是 agent 能直接处理的格式。
        result = self._api("getUpdates", offset=self._offset, timeout=30,
                           allowed_updates=["message"])
        if not result or not isinstance(result, list):
            return self._flush_all()

        for update in result:
            uid = update.get("update_id", 0)
            # 成功看到 update 后立刻推进 offset。下次轮询从 uid + 1 开始。
            if uid >= self._offset:
                self._offset = uid + 1
                save_offset(self._offset_path, self._offset)
            if uid in self._seen:
                continue
            self._seen.add(uid)
            if len(self._seen) > 5000:
                self._seen.clear()

            msg = update.get("message")
            if not msg:
                continue
            if msg.get("media_group_id"):
                # 多图/相册先不交给 agent, 等同组消息到齐后再合并。
                self._buf_media(msg, update)
                continue
            # _parse() 是 Telegram 原始字段 -> InboundMessage 的关键翻译步骤。
            inbound = self._parse(msg, update)
            if not inbound:
                continue
            if self.allowed_chats and inbound.peer_id not in self.allowed_chats:
                # 白名单过滤发生在归一化之后, 因为不同平台的 chat/user 字段名不同。
                continue
            self._buf_text(inbound)

        return self._flush_all()

    def _flush_all(self) -> list[InboundMessage]:
        ready = self._flush_media()
        ready.extend(self._flush_text())
        return ready

    # -- 媒体组缓冲 (500ms 窗口) --

    def _buf_media(self, msg: dict, update: dict) -> None:
        mgid = msg["media_group_id"]
        if mgid not in self._media_groups:
            self._media_groups[mgid] = {"ts": time.monotonic(), "entries": []}
        # 这里只保存平台原始 message, 真正的 InboundMessage 等 _flush_media() 再生成。
        self._media_groups[mgid]["entries"].append((msg, update))

    def _flush_media(self) -> list[InboundMessage]:
        now = time.monotonic()
        ready: list[InboundMessage] = []
        expired = [k for k, g in self._media_groups.items() if (now - g["ts"]) >= 0.5]
        for mgid in expired:
            entries = self._media_groups.pop(mgid)["entries"]
            captions, media_items = [], []
            for m, _ in entries:
                if m.get("caption"):
                    captions.append(m["caption"])
                for mt in ("photo", "video", "document", "audio"):
                    if mt in m:
                        raw_m = m[mt]
                        # Telegram photo 是多个尺寸组成的 list, 其他媒体多是 dict。
                        # 这里取可用于后续下载/引用的 file_id, 不在本节下载二进制内容。
                        fid = raw_m[-1]["file_id"] if isinstance(raw_m, list) and raw_m else raw_m.get("file_id", "") if isinstance(raw_m, dict) else ""
                        media_items.append({"type": mt, "file_id": fid})
            inbound = self._parse(entries[0][0], entries[0][1])
            if inbound:
                inbound.text = "\n".join(captions) if captions else "[media group]"
                inbound.media = media_items
                if not self.allowed_chats or inbound.peer_id in self.allowed_chats:
                    ready.append(inbound)
        return ready

    # -- 文本合并 (1s 窗口) --
    # Telegram 会将长粘贴拆分成多个片段; 缓冲后在 1s 静默后发出。

    def _buf_text(self, inbound: InboundMessage) -> None:
        key = (inbound.peer_id, inbound.sender_id)
        now = time.monotonic()
        if key in self._text_buf:
            self._text_buf[key]["text"] += "\n" + inbound.text
            self._text_buf[key]["ts"] = now
        else:
            self._text_buf[key] = {"text": inbound.text, "msg": inbound, "ts": now}

    def _flush_text(self) -> list[InboundMessage]:
        now = time.monotonic()
        ready: list[InboundMessage] = []
        expired = [k for k, b in self._text_buf.items() if (now - b["ts"]) >= 1.0]
        for key in expired:
            buf = self._text_buf.pop(key)
            buf["msg"].text = buf["text"]
            ready.append(buf["msg"])
        return ready

    # -- 消息解析 --

    def _parse(self, msg: dict, raw_update: dict) -> InboundMessage | None:
        # Telegram message 的关键结构大致是:
        # {
        #   "chat": {"id": -100..., "type": "supergroup"},
        #   "from": {"id": 123...},
        #   "text": "hello"
        # }
        #
        # 本函数把这些 Telegram 专属字段翻译成通用字段:
        #   chat.id   -> 群聊/频道会话 id
        #   from.id   -> 发送者 id
        #   text/caption -> 用户输入文本
        #
        # 翻译完成后, 后面的 agent 代码就不需要 import Telegram SDK,
        # 也不需要知道 update/message/chat/from 这些平台字段名。
        chat = msg.get("chat", {})
        chat_type = chat.get("type", "")
        chat_id = str(chat.get("id", ""))
        user_id = str(msg.get("from", {}).get("id", ""))
        text = msg.get("text", "") or msg.get("caption", "")
        if not text:
            return None

        thread_id = msg.get("message_thread_id")
        is_forum = chat.get("is_forum", False)
        is_group = chat_type in ("group", "supergroup")

        if chat_type == "private":
            # 私聊: 回复目标就是用户本人。
            peer_id = user_id
        elif is_group and is_forum and thread_id is not None:
            # Telegram forum topic: 回复时必须同时带 chat_id 和 message_thread_id。
            # 所以 peer_id 编码成 "chat_id:topic:thread_id", send() 再拆回来。
            peer_id = f"{chat_id}:topic:{thread_id}"
        else:
            # 普通群聊/超级群: 回复目标是 chat_id。
            peer_id = chat_id

        return InboundMessage(
            text=text, sender_id=user_id, channel="telegram",
            account_id=self.account_id, peer_id=peer_id,
            is_group=is_group, raw=raw_update,
        )

    def receive(self) -> InboundMessage | None:
        msgs = self.poll()
        return msgs[0] if msgs else None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        # run_agent_turn() 会调用 ch.send(inbound.peer_id, reply)。
        # 对 Telegram 来说, send() 要把通用 peer_id 还原成 Bot API 需要的参数。
        chat_id, thread_id = to, None
        if ":topic:" in to:
            parts = to.split(":topic:")
            chat_id, thread_id = parts[0], int(parts[1]) if len(parts) > 1 else None
        ok = True
        for chunk in self._chunk(text):
            # 这里是真正把回复发回 Telegram 的地方:
            # POST /bot<TOKEN>/sendMessage
            #   chat_id = 私聊 user_id 或群聊 chat_id
            #   message_thread_id = 话题 id, 仅 forum topic 需要
            if not self._api("sendMessage", chat_id=chat_id, text=chunk,
                             message_thread_id=thread_id):
                ok = False
        return ok

    def _chunk(self, text: str) -> list[str]:
        # Telegram 单条文本消息有长度上限。agent 输出过长时拆成多条发。
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text); break
            cut = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if cut <= 0:
                cut = self.MAX_MSG_LEN
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# FeishuChannel -- 基于 webhook (飞书/Lark)
# ---------------------------------------------------------------------------

class FeishuChannel(Channel):
    name = "feishu"

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("FeishuChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        # 飞书/Lark 不是用 bot token 直接发所有请求, 而是用 app_id/app_secret
        # 换取 tenant_access_token, 再用 tenant_access_token 调消息 API。
        self.app_id = account.config.get("app_id", "")
        self.app_secret = account.config.get("app_secret", "")
        # encrypt_key 在生产 webhook 中通常用于校验/解密事件。
        # 本教学版只做简单 token 对比, 没有完整实现飞书加密事件解密。
        self._encrypt_key = account.config.get("encrypt_key", "")
        # bot_open_id 用于群聊 @ 检测。没有 @ bot 的群消息通常不应该触发 agent。
        self._bot_open_id = account.config.get("bot_open_id", "")
        is_lark = account.config.get("is_lark", False)
        self.api_base = ("https://open.larksuite.com/open-apis" if is_lark
                         else "https://open.feishu.cn/open-apis")
        self._tenant_token: str = ""
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _refresh_token(self) -> str:
        # 飞书发送消息前需要 tenant_access_token。
        # token 有过期时间, 所以这里做简单缓存, 快过期时再刷新。
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        try:
            resp = self._http.post(
                f"{self.api_base}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Token error: {data.get('msg', '?')}{RESET}")
                return ""
            self._tenant_token = data.get("tenant_access_token", "")
            self._token_expires_at = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        except Exception as exc:
            print(f"  {RED}[feishu] Token error: {exc}{RESET}")
            return ""

    def _bot_mentioned(self, event: dict) -> bool:
        # 飞书群聊消息会带 mentions。这里检查消息是否 @ 了当前机器人。
        # 如果不做这层过滤, bot 可能会回复群里的每一句普通聊天。
        for m in event.get("message", {}).get("mentions", []):
            mid = m.get("id", {})
            if isinstance(mid, dict) and mid.get("open_id") == self._bot_open_id:
                return True
            if isinstance(mid, str) and mid == self._bot_open_id:
                return True
            if m.get("key") == self._bot_open_id:
                return True
        return False

    def _parse_content(self, message: dict) -> tuple[str, list]:
        # 飞书 message.content 是一个 JSON 字符串, 不同 msg_type 的结构不同。
        # 本函数只把常见类型抽成 agent 能处理的 text + media。
        msg_type = message.get("msg_type", "text")
        raw = message.get("content", "{}")
        try:
            content = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return "", []

        media: list[dict] = []
        if msg_type == "text":
            # 文本消息 content 形如 {"text": "..."}。
            return content.get("text", ""), media
        if msg_type == "post":
            # 富文本消息可能包含标题、普通文本、链接等节点。
            # 教学版把可读内容拍平成纯文本, 方便直接交给 LLM。
            texts: list[str] = []
            for lc in content.values():
                if not isinstance(lc, dict):
                    continue
                title = lc.get("title", "")
                if title:
                    texts.append(title)
                for para in lc.get("content", []):
                    for node in para:
                        tag = node.get("tag")
                        if tag == "text":
                            texts.append(node.get("text", ""))
                        elif tag == "a":
                            texts.append(node.get("text", "") + " " + node.get("href", ""))
            return "\n".join(texts), media
        if msg_type == "image":
            # 图片消息只保留 image_key。真正下载图片需要另调飞书文件接口。
            key = content.get("image_key", "")
            if key:
                media.append({"type": "image", "key": key})
            return "[image]", media
        return "", media

    def parse_event(self, payload: dict, token: str = "") -> InboundMessage | None:
        """
        解析飞书事件回调。使用简单的 token 校验进行验证。

        这就是飞书侧的“收消息”入口:
        1. 用户在飞书里给机器人发消息或在群里 @ 机器人。
        2. 飞书开放平台把事件 JSON POST 到你配置的 webhook 地址。
        3. 外部 HTTP 服务拿到 payload 后调用 channel.parse_event(payload)。
        4. parse_event() 把飞书字段翻译成 InboundMessage。

        Telegram 示例里 receive() 主动 getUpdates; 飞书则是平台主动推送。
        两者入口不同, 但最终都会变成同一个 InboundMessage。
        """
        if self._encrypt_key and token and token != self._encrypt_key:
            print(f"  {RED}[feishu] Token verification failed{RESET}")
            return None
        if "challenge" in payload:
            print_info(f"[feishu] Challenge: {payload['challenge']}")
            return None

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("open_id", sender.get("user_id", ""))
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"

        if is_group and self._bot_open_id and not self._bot_mentioned(event):
            # 群聊里没有 @ 当前机器人就忽略。
            return None

        text, media = self._parse_content(message)
        if not text:
            return None

        # 飞书字段 -> 通用字段:
        #   sender.sender_id.open_id -> sender_id
        #   message.chat_id          -> 群聊会话 id
        #   message.chat_type        -> p2p/group
        #   message.content          -> text/media
        #
        # peer_id 是“回复目标”。这里为了让会话按用户隔离, p2p 使用 user_id;
        # 群聊使用 chat_id。正式接入时也可以把 receive_id_type 一并放到 raw/config,
        # 这样 send() 能区分 open_id/user_id/chat_id 等飞书发送目标类型。
        return InboundMessage(
            text=text, sender_id=user_id, channel="feishu",
            account_id=self.account_id,
            peer_id=user_id if chat_type == "p2p" else chat_id,
            media=media, is_group=is_group, raw=payload,
        )

    def receive(self) -> InboundMessage | None:
        # 飞书是 webhook 推送模型, 本文件没有启动 HTTP server,
        # 所以 receive() 没有东西可轮询。真实服务会在 webhook handler 中调用 parse_event()。
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        # 飞书发消息需要先拿 tenant_access_token, 然后调用 im/v1/messages。
        # 这里的 to 来自 inbound.peer_id。教学版固定用 receive_id_type=chat_id,
        # 所以群聊回复路径最直观; 如果要完整支持飞书单聊, 应把 p2p 的发送目标
        # 明确保存为 chat_id 或 open_id 并对应调整 receive_id_type。
        token = self._refresh_token()
        if not token:
            return False
        try:
            resp = self._http.post(
                f"{self.api_base}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": to, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Send: {data.get('msg', '?')}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"  {RED}[feishu] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
MEMORY_FILE = WORKSPACE_DIR / "MEMORY.md"

def tool_memory_write(content: str) -> str:
    print_tool("memory_write", f"{len(content)} chars")
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n- {content}\n")
        return f"Written to memory: {content[:80]}..."
    except Exception as exc:
        return f"Error: {exc}"

def tool_memory_search(query: str) -> str:
    print_tool("memory_search", query)
    if not MEMORY_FILE.exists():
        return "Memory file is empty."
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        matches = [l for l in lines if query.lower() in l.lower()]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."
    except Exception as exc:
        return f"Error: {exc}"

TOOLS = [
    {"name": "memory_write", "description": "Save a note to long-term memory.",
     "input_schema": {"type": "object", "required": ["content"],
                      "properties": {"content": {"type": "string",
                                                  "description": "The text to remember."}}}},
    {"name": "memory_search", "description": "Search through saved memory notes.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string",
                                               "description": "Search keyword."}}}},
]

TOOL_HANDLERS: dict[str, Any] = {
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"

# ---------------------------------------------------------------------------
# ChannelManager
# ---------------------------------------------------------------------------

class ChannelManager:
    def __init__(self) -> None:
        # channels 是“平台名 -> 通道实例”的注册表。
        # run_agent_turn() 只知道 inbound.channel, 通过这里找到对应的发送器。
        self.channels: dict[str, Channel] = {}
        # accounts 保存机器人账号配置, 主要给 /accounts 命令展示。
        self.accounts: list[ChannelAccount] = []

    def register(self, channel: Channel) -> None:
        # 例如注册后:
        #   self.channels["cli"] = CLIChannel(...)
        #   self.channels["telegram"] = TelegramChannel(...)
        # 后续 ch = mgr.get(inbound.channel) 就能把回复发回原平台。
        self.channels[channel.name] = channel
        print_channel(f"  [+] Channel registered: {channel.name}")

    def list_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def close_all(self) -> None:
        for ch in self.channels.values():
            ch.close()

# ---------------------------------------------------------------------------
# Telegram 后台轮询线程
# ---------------------------------------------------------------------------

def telegram_poll_loop(
    tg: TelegramChannel, queue: list, lock: threading.Lock, stop: threading.Event,
) -> None:
    print_channel(f"  [telegram] Polling started for {tg.account_id}")
    while not stop.is_set():
        try:
            # 后台线程不断从 Telegram 拉消息, poll() 返回的已经是 InboundMessage。
            # 主线程只需要从 queue 取出来交给 run_agent_turn()。
            msgs = tg.poll()
            if msgs:
                with lock:
                    queue.extend(msgs)
        except Exception as exc:
            print(f"  {RED}[telegram] Poll error: {exc}{RESET}")
            stop.wait(5.0)

# ---------------------------------------------------------------------------
# REPL 命令
# ---------------------------------------------------------------------------

def handle_repl_command(cmd: str, mgr: ChannelManager) -> bool:
    cmd = cmd.strip().lower()
    if cmd == "/channels":
        for name in mgr.list_channels():
            print_channel(f"  - {name}")
        return True
    if cmd == "/accounts":
        for acc in mgr.accounts:
            masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
            print_channel(f"  - {acc.channel}/{acc.account_id}  token={masked}")
        return True
    if cmd in ("/help", "/h"):
        print_info("  /channels  /accounts  /help  quit/exit")
        return True
    return False

# ---------------------------------------------------------------------------
# Agent 回合
# ---------------------------------------------------------------------------

def run_agent_turn(
    inbound: InboundMessage,
    conversations: dict[str, list[dict]],
    mgr: ChannelManager,
) -> None:
    # 从这里开始, agent 逻辑已经完全和社交媒体平台解耦。
    # inbound 可能来自 Telegram、飞书、CLI, 但函数只读取统一字段:
    #   inbound.text    -> 放进 LLM messages
    #   inbound.channel -> 找到原通道
    #   inbound.peer_id -> 把回复发回原会话
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    if inbound.channel == "telegram":
        # 这是一个可选的平台体验优化。只有 Telegram 支持 sendChatAction,
        # 所以这里特判一下; 核心 agent 流程仍然不依赖 Telegram。
        tg = mgr.get("telegram")
        if isinstance(tg, TelegramChannel):
            tg.send_typing(inbound.peer_id.split(":topic:")[0])

    while True:
        try:
            response = client.messages.create(
                model=MODEL_ID, max_tokens=8096,
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                # 回复的关键路径:
                # 1. inbound.channel 告诉我们消息来自哪个平台。
                # 2. ChannelManager 找到对应 Channel。
                # 3. inbound.peer_id 告诉 Channel 发回哪个用户/群/话题。
                #
                # 也就是说, agent 不直接调用 Telegram/飞书 API;
                # 它只把文本交还给“原来的通道”。
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
                else:
                    print_assistant(text, inbound.channel)
            break
        elif response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": process_tool_call(block.name, block.input),
                    })
            messages.append({"role": "user", "content": results})
        else:
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
            break

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    mgr = ChannelManager()
    cli = CLIChannel()
    # CLI 永远注册, 这样即使没有配置任何社交媒体 token, 也能在终端试完整流程。
    mgr.register(cli)

    tg_channel: TelegramChannel | None = None
    stop_event = threading.Event()
    msg_queue: list[InboundMessage] = []
    q_lock = threading.Lock()
    tg_thread: threading.Thread | None = None

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if tg_token and HAS_HTTPX:
        # 只要 .env 里有 TELEGRAM_BOT_TOKEN, 本程序就会创建 TelegramChannel。
        # 这一步相当于“把 Telegram bot 接到 agent 上”。
        tg_acc = ChannelAccount(
            channel="telegram", account_id="tg-primary", token=tg_token,
            config={"allowed_chats": os.getenv("TELEGRAM_ALLOWED_CHATS", "")},
        )
        mgr.accounts.append(tg_acc)
        tg_channel = TelegramChannel(tg_acc)
        mgr.register(tg_channel)
        tg_thread = threading.Thread(
            target=telegram_poll_loop, daemon=True,
            args=(tg_channel, msg_queue, q_lock, stop_event),
        )
        tg_thread.start()

    fs_id = os.getenv("FEISHU_APP_ID", "").strip()
    fs_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if fs_id and fs_secret and HAS_HTTPX:
        # 只要 .env 里有飞书 app_id/app_secret, 就注册 FeishuChannel。
        # 注意: 本文件没有启动 webhook HTTP server, 所以这里只展示解析和发送能力;
        # 真实接入需要让飞书开放平台把事件 POST 到你的服务, 服务再调用 parse_event()。
        fs_acc = ChannelAccount(
            channel="feishu", account_id="feishu-primary",
            config={
                "app_id": fs_id, "app_secret": fs_secret,
                "encrypt_key": os.getenv("FEISHU_ENCRYPT_KEY", ""),
                "bot_open_id": os.getenv("FEISHU_BOT_OPEN_ID", ""),
                "is_lark": os.getenv("FEISHU_IS_LARK", "").lower() in ("1", "true"),
            },
        )
        mgr.accounts.append(fs_acc)
        mgr.register(FeishuChannel(fs_acc))

    print_info("=" * 60)
    print_info("  claw0  |  Section 04: Channels")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Channels: {', '.join(mgr.list_channels())}")
    print_info("  Commands: /channels /accounts /help  |  quit/exit")
    print_info("=" * 60)
    print()

    conversations: dict[str, list[dict]] = {}

    while True:
        # 排空 Telegram 队列
        with q_lock:
            tg_msgs = msg_queue[:]
            msg_queue.clear()
        for m in tg_msgs:
            # m 已经是从 Telegram update 归一化来的 InboundMessage。
            print_channel(f"\n  [telegram] {m.sender_id}: {m.text[:80]}")
            run_agent_turn(m, conversations, mgr)

        # CLI 输入 (当 Telegram 活跃时使用非阻塞模式)
        if tg_channel:
            import select
            if not select.select([sys.stdin], [], [], 0.5)[0]:
                continue
            try:
                user_input = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
        else:
            msg = cli.receive()
            if msg is None:
                break
            user_input = msg.text

        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.startswith("/") and handle_repl_command(user_input, mgr):
            continue

        run_agent_turn(
            # CLI 输入也包装成 InboundMessage, 所以后面的处理路径和 Telegram 完全一样。
            InboundMessage(text=user_input, sender_id="cli-user",
                           channel="cli", account_id="cli-local", peer_id="cli-user"),
            conversations, mgr,
        )

    print(f"{DIM}Goodbye.{RESET}")
    stop_event.set()
    if tg_thread and tg_thread.is_alive():
        tg_thread.join(timeout=3.0)
    mgr.close_all()

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    agent_loop()

if __name__ == "__main__":
    main()

"""
 > 我们不是让 Agent 分别理解 Telegram、飞书、CLI。我们在外面加了一层 Channel 适配器。每个平台的消息先被转换成统一的 InboundMessage，Agent 只处理这个统一格式。回复时，再根据消息来源找到对应的 Channel，把回复
  > 发回原平台。                                                                                                                                                                                               

  更具体一点：

  Telegram update     \
  飞书 webhook payload  ->  InboundMessage  ->  Agent Loop  ->  Channel.send()
  CLI input           /

  也就是说，平台差异被关在 Channel 里面。

  比如 Telegram：

  Telegram getUpdates
      -> TelegramChannel._parse()
      -> InboundMessage(channel="telegram", peer_id=chat_id, text="...")
      -> run_agent_turn()
      -> TelegramChannel.send()
      -> Telegram sendMessage API

  飞书类似：

  飞书 webhook 事件
      -> FeishuChannel.parse_event()
      -> InboundMessage(channel="feishu", peer_id=chat_id, text="...")
      -> run_agent_turn()
      -> FeishuChannel.send()
      -> 飞书 im/v1/messages API

  所以别人问“你们怎么做到对接多个平台的？”你可以回答：

  > 核心是适配器模式。我们定义了一个统一的 Channel 接口，每个平台只需要实现两件事：receive() 把平台消息转成统一格式，send() 把 Agent 回复发回平台。Agent 本身不关心消息来自 Telegram、飞书还是命令行。         

  这份 sessions/zh/s04_channels.py:1 是最小教学版。Telegram 部分配置 token 后是真正能通过 Bot API 拉消息和发消息的；飞书部分展示了解析 webhook 和发送消息的适配逻辑，但这个文件本身没有启动 HTTP webhook
  server，所以还不是完整线上飞书接入。完整生产版还要加鉴权、重试、限流、持久化、路由等。
"""
