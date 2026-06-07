r"""
Section 06: Intelligence (智能)
"赋予灵魂, 教会记忆"

每轮对话前, agent 的"大脑"是如何组装的?
本节是整个教学项目的核心集成点 -- 演示系统提示词的分层构建过程.

在 s01-s02 中, 系统提示词是硬编码的字符串.
在真实的 agent 框架中, 系统提示词由多个层级动态组装:
  Identity / 灵魂 / Tools / 技能 / Memory / Bootstrap / Runtime / Channel

学习路线:
  1. 先看 BootstrapLoader: agent 启动时从磁盘读哪些"自我说明".
  2. 再看 SkillsManager: 为什么技能不是写死在代码里, 而是动态发现.
  3. 重点看 MemoryStore: 记忆如何写入、切块、检索、重排.
  4. 最后看 build_system_prompt() 和 agent_loop(): 每轮对话前,
     如何把身份、人格、工具、技能、记忆、运行时上下文组装成一次 LLM 调用.

本文件的几个亮点:
  - "提示词不是一个字符串": 它是由 8 层上下文拼出来的运行时产物.
  - "记忆不是聊天记录": 这里把长期事实和每日事件分开存储, 再按问题召回.
  - "检索不是只搜关键词": 混合搜索同时演示关键词、向量、时间衰减和 MMR 多样性.
  - "工具不是 Python 函数直接暴露": 需要 schema 给模型看, handler 给程序执行.
  - "每轮都重建系统提示词": 因为上一轮可能写入了新记忆, 下一轮就应该能用上.

架构:

    [SOUL.md]  [IDENTITY.md]  [TOOLS.md]  [MEMORY.md]  ...
         \          |            |           /
          v         v            v          v
        +-------------------------------+
        |     BootstrapLoader           |
        |  (load, truncate, cap)        |
        +-------------------------------+
                    |
                    v
        +-------------------------------+        +-------------------+
        |   build_system_prompt()       | <----> | SkillsManager     |
        |   (8 层组装)                  |        | (discover, parse) |
        +-------------------------------+        +-------------------+
                    |                                     ^
                    v                                     |
        +-------------------------------+        +-------------------+
        |   Agent Loop (每轮)           | <----> | MemoryStore       |
        |   search -> build -> call LLM |        | (write, search)   |
        +-------------------------------+        +-------------------+

用法:
    cd claw0
    python zh/s06_intelligence.py

REPL 命令:
    /soul /skills /memory /search <q> /prompt /bootstrap
"""

# ---------------------------------------------------------------------------
# 导入与配置
# ---------------------------------------------------------------------------
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

# .env 位于项目根目录. 这里用 Path(__file__) 反推根目录,
# 好处是无论你从哪个当前目录启动脚本, 都能找到同一个配置文件.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# MODEL_ID 和 client 是所有 LLM 调用的公共配置.
# 生产系统里通常会把不同 agent / 不同任务映射到不同模型;
# 教学版先保持一个全局模型, 方便聚焦"上下文怎么组装".
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# WORKSPACE_DIR 是 agent 的"家目录": Bootstrap 文件、技能、记忆都从这里找.
# 这和普通 Python 项目里的代码目录不同: workspace 更像运行时配置和状态目录.
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

# Bootstrap 文件名 -- 每个 agent 启动时加载这 8 个文件
# 亮点:
#   这些文件不是同一种东西. 它们分别承担身份、人格、工具规范、用户信息、
#   心跳/后台任务、启动规则、项目约束和长期记忆. 分文件的好处是可替换、
#   可覆盖、可单独截断, 不需要把所有规则硬编码进 Python.
BOOTSTRAP_FILES = [
    "SOUL.md", "IDENTITY.md", "TOOLS.md", "USER.md",
    "HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "MEMORY.md",
]

# 这些上限体现了真实 agent 框架里很重要的工程约束:
#   1. LLM 上下文窗口有限, 不能把磁盘上的所有东西一股脑塞进去.
#   2. 单个文件可能异常巨大, 所以既有单文件上限, 也有总上限.
#   3. 技能太多会稀释注意力, 所以技能数量和技能提示词也要封顶.
MAX_FILE_CHARS = 20000
MAX_TOTAL_CHARS = 150000
MAX_SKILLS = 150
MAX_SKILLS_PROMPT = 30000

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"


def colored_prompt() -> str:
    # 只负责 REPL 展示, 不参与 agent 智能逻辑.
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    # 工具调用单独用暗色打印, 方便你观察模型什么时候选择了工具.
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_section(title: str) -> None:
    print(f"\n{MAGENTA}{BOLD}--- {title} ---{RESET}")


# ---------------------------------------------------------------------------
# 1. Bootstrap 文件加载器
# ---------------------------------------------------------------------------
# 在 agent 启动时加载工作区的 Bootstrap 文件.
# 不同加载模式 (full/minimal/none) 适用于不同场景:
#   full = 主 agent | minimal = 子 agent / cron | none = 最小化

class BootstrapLoader:
    """
    BootstrapLoader 是"磁盘配置 -> prompt 上下文"的第一道关卡.

    学习重点:
      - 它不理解文件内容的语义, 只负责安全加载和截断.
      - 真正决定这些内容放到 prompt 哪一层的是 build_system_prompt().
      - 缺文件时返回空字符串, 让 agent 能在不完整工作区里继续启动.
    """

    def __init__(self, workspace_dir: Path) -> None:
        # 只保存 workspace 根目录, 之后所有 Bootstrap 文件都基于它拼路径.
        # 这样不会受当前终端目录变化影响.
        self.workspace_dir = workspace_dir

    def load_file(self, name: str) -> str:
        # name 来自 BOOTSTRAP_FILES 这种白名单, 不是用户随便输入的路径.
        # 真实系统里也应该避免让模型/用户直接控制可读取的文件名.
        path = self.workspace_dir / name
        if not path.is_file():
            return ""
        try:
            # Markdown 文档统一按 utf-8 读取. 失败时吞掉异常, 是为了让某个坏文件
            # 不至于拖垮整个 agent 启动流程.
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def truncate_file(self, content: str, max_chars: int = MAX_FILE_CHARS) -> str:
        """截断超长文件内容. 仅保留头部, 在行边界处截断."""
        if len(content) <= max_chars:
            return content
        # 尽量在换行处截断, 这样不会把一行 Markdown 或 JSON 切在中间.
        # 这是一个小细节, 但对 prompt 可读性很重要.
        cut = content.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        return content[:cut] + f"\n\n[... truncated ({len(content)} chars total, showing first {cut}) ...]"

    def load_all(self, mode: str = "full") -> dict[str, str]:
        # none 模式完全不加载 Bootstrap. 适合极简测试、低成本任务,
        # 或者你只想观察没有任何额外上下文时模型的基础行为.
        if mode == "none":
            return {}
        # minimal 模式只加载项目约束和工具规范. 这是给子 agent / 定时任务用的:
        # 它们不需要完整人格和记忆, 但仍然应该遵守 AGENTS.md 和工具约束.
        names = ["AGENTS.md", "TOOLS.md"] if mode == "minimal" else list(BOOTSTRAP_FILES)
        result: dict[str, str] = {}
        total = 0
        for name in names:
            raw = self.load_file(name)
            if not raw:
                continue
            truncated = self.truncate_file(raw)
            if total + len(truncated) > MAX_TOTAL_CHARS:
                # 总量上限比单文件上限更重要: 它保护整次 LLM 调用的上下文预算.
                # 如果已经快满了, 就只塞入剩余能容纳的部分.
                remaining = MAX_TOTAL_CHARS - total
                if remaining > 0:
                    truncated = self.truncate_file(raw, remaining)
                else:
                    break
            result[name] = truncated
            total += len(truncated)
        # 返回 dict 的原因: 后面 build_system_prompt() 需要按文件名决定层级,
        # 例如 SOUL.md 放人格层, MEMORY.md 放记忆层.
        return result


# ---------------------------------------------------------------------------
# 2. 灵魂系统
# ---------------------------------------------------------------------------
# SOUL.md 定义 agent 的人格. 不同 agent 可以有不同的 SOUL.md 文件.
# 注入到系统提示词的靠前位置 -- 越靠前影响力越强.


def load_soul(workspace_dir: Path) -> str:
    # 这个函数保留为独立 helper, 是为了强调 SOUL.md 的概念:
    # 它不是普通配置, 而是 agent 的人格/语气/行为偏好.
    # 当前主流程通过 BootstrapLoader 一次性加载, 但单独函数方便教学和扩展.
    path = workspace_dir / "SOUL.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. 技能发现与注入
# ---------------------------------------------------------------------------
# 一个技能 = 一个包含 SKILL.md (带 frontmatter) 的目录.
# 按优先级顺序扫描; 同名技能会被后发现的覆盖.


class SkillsManager:
    """
    技能管理器负责把"文件系统里的技能"变成"模型能看到的技能说明".

    亮点:
      - 技能不是 Python import, 而是 prompt 能读懂的一段说明.
      - 每个技能目录只要求有 SKILL.md, 方便插件化和用户自定义.
      - 同名技能后扫描的覆盖先扫描的, 等价于"更局部的配置覆盖全局配置".
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        # 每个元素保存 name/description/invocation/body/path.
        # 这里用 dict 是为了教学简洁; 生产代码里可以换成 dataclass.
        self.skills: list[dict[str, str]] = []

    def _parse_frontmatter(self, text: str) -> dict[str, str]:
        """解析简单的 YAML frontmatter, 不依赖 pyyaml."""
        meta: dict[str, str] = {}
        # SKILL.md 约定以 --- 开头:
        #   ---
        #   name: xxx
        #   description: xxx
        #   invocation: xxx
        #   ---
        # 教学版只解析 key: value 这种一行形式, 避免引入 YAML 依赖.
        if not text.startswith("---"):
            return meta
        parts = text.split("---", 2)
        if len(parts) < 3:
            return meta
        for line in parts[1].strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            meta[key.strip()] = value.strip()
        return meta

    def _scan_dir(self, base: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        if not base.is_dir():
            return found
        # sorted() 保证扫描顺序稳定. 对教学和调试很重要:
        # 同样的目录结构每次得到同样的技能顺序.
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta = self._parse_frontmatter(content)
            if not meta.get("name"):
                # 没有 name 的技能无法被去重和展示, 因此直接跳过.
                continue
            body = ""
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # frontmatter 后面的正文才是模型真正要阅读的操作说明.
                    body = parts[2].strip()
            found.append({
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "invocation": meta.get("invocation", ""),
                "body": body,
                "path": str(child),
            })
        return found

    def discover(self, extra_dirs: list[Path] | None = None) -> None:
        """按优先级扫描技能目录; 同名技能后者覆盖前者."""
        scan_order: list[Path] = []
        if extra_dirs:
            # extra_dirs 放在最前面, 但由于后发现覆盖先发现,
            # 它适合做"基础技能源"; 后面的 workspace/project 可以覆盖它.
            scan_order.extend(extra_dirs)
        scan_order.append(self.workspace_dir / "skills")           # 内置技能
        scan_order.append(self.workspace_dir / ".skills")          # 托管技能
        scan_order.append(self.workspace_dir / ".agents" / "skills")  # 个人 agent 技能
        scan_order.append(Path.cwd() / ".agents" / "skills")      # 项目 agent 技能
        scan_order.append(Path.cwd() / "skills")                  # 工作区技能

        seen: dict[str, dict[str, str]] = {}
        for d in scan_order:
            for skill in self._scan_dir(d):
                # 同名覆盖是这个函数最关键的设计点:
                # 越靠近当前项目/当前 agent 的技能, 越应该有更高优先级.
                seen[skill["name"]] = skill
        # 限制技能数量, 避免 prompt 被大量技能挤爆.
        self.skills = list(seen.values())[:MAX_SKILLS]

    def format_prompt_block(self) -> str:
        # 这里把技能列表格式化成系统提示词的一整块.
        # 注意: 工具 TOOLS 是给 API 的结构化 schema;
        # 技能 skills_block 是给模型阅读的自然语言操作手册.
        if not self.skills:
            return ""
        lines = ["## Available Skills", ""]
        total = 0
        for skill in self.skills:
            block = (
                f"### Skill: {skill['name']}\n"
                f"Description: {skill['description']}\n"
                f"Invocation: {skill['invocation']}\n"
            )
            if skill.get("body"):
                block += f"\n{skill['body']}\n"
            block += "\n"
            if total + len(block) > MAX_SKILLS_PROMPT:
                # 技能提示词也有独立预算. 技能太多时, 宁可截断,
                # 也不要让核心身份/记忆/项目约束被挤出上下文.
                lines.append(f"(... more skills truncated)")
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 记忆系统
# ---------------------------------------------------------------------------
# 两层存储:
#   MEMORY.md = 长期事实 (手动维护)
#   daily/{date}.jsonl = 每日日志 (通过 agent 工具自动写入)
# 搜索: TF-IDF + 余弦相似度, 纯 Python 实现


class MemoryStore:
    """
    MemoryStore 演示一个最小但完整的 agent 记忆系统.

    它故意不用数据库和 embedding API, 因为本节要先讲清楚"形状":
      - 记忆写在哪里.
      - 怎样把长文档拆成可检索的块.
      - 怎样根据当前用户输入召回相关片段.
      - 怎样把召回结果注入系统提示词.

    真实系统可以把 JSONL 换成 SQLite/Postgres, 把哈希向量换成 embedding,
    但写入、检索、重排、注入这条主线是不变的.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        # daily 目录保存 agent 运行过程中自动写入的事件型记忆.
        # 目录不存在就创建, 这样第一次运行 / memory_write 时不会失败.
        self.memory_dir = workspace_dir / "memory" / "daily"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(self, content: str, category: str = "general") -> str:
        # 每天一个 JSONL 文件:
        #   - 方便追加写入, 不需要读完整文件再改.
        #   - 方便按日期做归档、清理、时间衰减.
        #   - 每行一条 JSON, 单行坏掉也不影响其它行.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.memory_dir / f"{today}.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "content": content,
        }
        try:
            # ensure_ascii=False 保留中文原文, 学习和调试时更容易读.
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return f"Memory saved to {today}.jsonl ({category})"
        except Exception as exc:
            return f"Error writing memory: {exc}"

    def load_evergreen(self) -> str:
        # MEMORY.md 是"常驻记忆": 稳定、长期、通常由人维护.
        # daily/*.jsonl 是"事件记忆": 运行时不断积累.
        # 这两类分开, 是为了避免临时聊天事件污染长期事实.
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self) -> list[dict[str, str]]:
        """加载所有记忆并拆分为块 (path + text)."""
        chunks: list[dict[str, str]] = []
        # 按段落拆分长期记忆.
        # 亮点:
        #   检索系统通常不是把整份 MEMORY.md 当一个文档.
        #   如果整份文档太大, 一个小事实命中时会把无关内容一起带进 prompt.
        #   按段落切块可以让召回结果更精确.
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        # 每日记忆: 每条 JSONL 记录作为一个块.
        # 一条 memory_write 通常就是一个独立事实/观察, 天然适合做 chunk.
        if self.memory_dir.is_dir():
            for jf in sorted(self.memory_dir.glob("*.jsonl")):
                try:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("content", "")
                        if text:
                            cat = entry.get("category", "")
                            label = f"{jf.name} [{cat}]" if cat else jf.name
                            chunks.append({"path": label, "text": text})
                except Exception:
                    # 某个 JSONL 文件损坏时跳过, 保持整体记忆系统可用.
                    # 生产系统会记录日志, 教学版只保留容错行为.
                    continue
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词: 小写英文 + 单个 CJK 字符, 过滤短 token."""
        # 英文/数字按连续串提取, 中文按 Unicode CJK 范围提取.
        # 这里不是专业中文分词, 但足以演示检索算法.
        #
        # 过滤规则:
        #   - 英文单字母通常噪声很大, 所以过滤.
        #   - 单个中文字符可能有意义, 所以保留.
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """TF-IDF + 余弦相似度搜索, 纯 Python 实现."""
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 先把每个记忆块分词. 后面会把 query 和 chunk 都转成向量.
        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

        # 文档频率 DF(document frequency):
        #   df[t] = 有多少个 chunk 出现过 token t.
        #
        # 直觉:
        #   "我", "今天", "项目" 这种到处出现的词区分度低.
        #   "Postgres", "蓝色", "过敏" 这种少见词更能帮助定位相关记忆.
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        n = len(chunks)

        def tfidf(tokens: list[str]) -> dict[str, float]:
            # TF(term frequency): token 在当前文本里出现多少次.
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # IDF(inverse document frequency): 越少见的 token 权重越高.
            # +1 是平滑项, 避免除零, 也避免极端小数据下权重过大.
            return {t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, c in tf.items()}

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            # 余弦相似度只关心方向, 不直接被文本长度支配.
            # 两段文本如果共享的高权重词多, 分数就高.
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored: list[dict[str, Any]] = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                snippet = chunks[i]["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                # 返回 path 是为了让用户知道这条记忆来自长期 MEMORY.md
                # 还是某个日期的 daily JSONL.
                scored.append({"path": chunks[i]["path"], "score": round(score, 4), "snippet": snippet})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # --- Hybrid Memory Search Enhancement ---

    @staticmethod
    def _hash_vector(text: str, dim: int = 64) -> list[float]:
        """Simulated vector embedding using hash-based random projection.
        No external API needed -- teaches the PATTERN of a second search channel."""
        # 亮点:
        #   这里不是生产级 embedding, 而是"模拟向量检索通道".
        #   目的在于让你看到混合搜索的结构:
        #       query/text -> vector -> cosine similarity -> ranked results
        #
        # 注意:
        #   Python 的 hash() 默认在不同进程间可能有随机种子,
        #   所以这个向量只适合教学演示, 不适合持久化或跨进程复现.
        #   生产系统会用 OpenAI / Voyage / 本地模型等 embedding.
        tokens = MemoryStore._tokenize(text)
        vec = [0.0] * dim
        for token in tokens:
            h = hash(token)
            for i in range(dim):
                # 用 hash 的每一位决定在该维度上 +1 还是 -1.
                # 多个 token 累加后, 得到一个固定维度的粗糙语义指纹.
                bit = (h >> (i % 62)) & 1
                vec[i] += 1.0 if bit else -1.0
        # 归一化后, 后面的点积/余弦相似度不会被 token 数量直接放大.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _vector_cosine(a: list[float], b: list[float]) -> float:
        # list[float] 版本的余弦相似度, 用于模拟向量检索.
        # 和上面 dict 版本的 TF-IDF cosine 是同一个数学思想.
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _bm25_rank_to_score(rank: int) -> float:
        """Convert BM25 rank position to a [0, 1] score."""
        # 这个 helper 预留给 BM25 排名融合思路:
        #   第 0 名 -> 1.0, 第 1 名 -> 0.5, 第 2 名 -> 0.333...
        # 当前教学版用 TF-IDF 分数直接融合, 所以它暂时没有被调用.
        return 1.0 / (1.0 + rank)

    @staticmethod
    def _jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        # Jaccard = 交集大小 / 并集大小.
        # 这里用它衡量两个候选记忆是否"内容太像", 给 MMR 去重/多样性使用.
        set_a, set_b = set(tokens_a), set(tokens_b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    def _vector_search(self, query: str, chunks: list[dict[str, str]], top_k: int = 10) -> list[dict[str, Any]]:
        """Search by simulated vector similarity."""
        # 向量通道擅长找"表达不完全一样, 但整体相近"的内容.
        # 在这个教学版里语义能力很弱, 但流程和真实向量库一致.
        q_vec = self._hash_vector(query)
        scored = []
        for chunk in chunks:
            c_vec = self._hash_vector(chunk["text"])
            score = self._vector_cosine(q_vec, c_vec)
            if score > 0.0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _keyword_search(self, query: str, chunks: list[dict[str, str]], top_k: int = 10) -> list[dict[str, Any]]:
        """Reuse existing TF-IDF as the keyword channel, return ranked results."""
        # 关键词通道擅长找"字面上命中"的内容.
        # 混合搜索通常不会只押注一种通道:
        #   - 关键词: 精确、可解释, 但同义改写容易漏.
        #   - 向量: 宽泛、召回强, 但可能带来语义漂移.
        # 两者合并可以互补.
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]
        n = len(chunks)
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, c in tf.items()}

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                scored.append({"chunk": chunks[i], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _merge_hybrid_results(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Merge vector and keyword results by weighted score combination."""
        # 融合策略:
        #   1. 先把向量结果放入 merged, 乘以 vector_weight.
        #   2. 再把关键词结果合并进去, 同一 chunk 分数相加.
        #   3. 最后按总分排序.
        #
        # 亮点:
        #   同时被两个通道命中的记忆会自然加分.
        #   这就是 hybrid search 的常见思想: 多个证据源互相确认.
        merged: dict[str, dict[str, Any]] = {}
        for r in vector_results:
            # 教学版用文本前 100 字做去重 key.
            # 生产系统应使用稳定 chunk_id, 否则两条前缀相同的记忆可能被误合并.
            key = r["chunk"]["text"][:100]
            merged[key] = {"chunk": r["chunk"], "score": r["score"] * vector_weight}
        for r in keyword_results:
            key = r["chunk"]["text"][:100]
            if key in merged:
                merged[key]["score"] += r["score"] * text_weight
            else:
                merged[key] = {"chunk": r["chunk"], "score": r["score"] * text_weight}
        result = list(merged.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    @staticmethod
    def _temporal_decay(results: list[dict[str, Any]], decay_rate: float = 0.01) -> list[dict[str, Any]]:
        """Apply exponential temporal decay to scores based on chunk age."""
        # 时间衰减解决一个真实问题:
        #   有些记忆越新越重要, 例如"用户现在正在做哪个项目".
        #   很久以前的相似记忆不应该总是压过最近的上下文.
        #
        # 公式:
        #   score *= exp(-decay_rate * age_days)
        # age_days 越大, 乘上的系数越小.
        now = datetime.now(timezone.utc)
        for r in results:
            path = r["chunk"].get("path", "")
            age_days = 0.0
            # daily 文件名里带日期, 例如 2026-06-07.jsonl [preference].
            # 从 path 中提取日期后即可估算记忆年龄.
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            if date_match:
                try:
                    chunk_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    age_days = (now - chunk_date).total_seconds() / 86400.0
                except ValueError:
                    pass
            r["score"] *= math.exp(-decay_rate * age_days)
        return results

    @staticmethod
    def _mmr_rerank(
        results: list[dict[str, Any]],
        lambda_param: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Maximal Marginal Relevance re-ranking for diversity.
        MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected"""
        # MMR 的价值:
        #   如果 top-5 全是同一句话的轻微改写, prompt 会浪费空间.
        #   MMR 会在"相关性"和"多样性"之间折中.
        #
        # lambda_param 越接近 1: 越重视相关性.
        # lambda_param 越接近 0: 越重视结果之间不要重复.
        if len(results) <= 1:
            return results
        # 先把所有候选结果分词, 后面用 Jaccard 估计候选之间的相似度.
        tokenized = [MemoryStore._tokenize(r["chunk"]["text"]) for r in results]
        selected: list[int] = []
        remaining = list(range(len(results)))
        reranked: list[dict[str, Any]] = []
        while remaining:
            best_idx = -1
            best_mmr = float("-inf")
            for idx in remaining:
                relevance = results[idx]["score"]
                max_sim = 0.0
                for sel_idx in selected:
                    sim = MemoryStore._jaccard_similarity(tokenized[idx], tokenized[sel_idx])
                    if sim > max_sim:
                        max_sim = sim
                # 相关性越高越好; 和已选结果越像, 惩罚越大.
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            reranked.append(results[best_idx])
        return reranked

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Full hybrid search pipeline: keyword -> vector -> merge -> decay -> MMR -> top_k"""
        # 这是记忆检索的总入口, 也是本节最值得记住的流水线:
        #   1. 加载记忆块
        #   2. 关键词召回
        #   3. 向量召回
        #   4. 加权融合
        #   5. 时间衰减
        #   6. MMR 重排
        #   7. 截取 top_k 给 prompt
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        keyword_results = self._keyword_search(query, chunks, top_k=10)
        vector_results = self._vector_search(query, chunks, top_k=10)
        merged = self._merge_hybrid_results(vector_results, keyword_results)
        decayed = self._temporal_decay(merged)
        reranked = self._mmr_rerank(decayed)
        result = []
        for r in reranked[:top_k]:
            snippet = r["chunk"]["text"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            # 对外只暴露 path/score/snippet, 不暴露内部 chunk 对象,
            # 这样工具输出和 prompt 注入都更稳定.
            result.append({"path": r["chunk"]["path"], "score": round(r["score"], 4), "snippet": snippet})
        return result

    def get_stats(self) -> dict[str, Any]:
        # REPL 的 /memory 命令使用这个统计信息.
        # 它不是智能核心, 但对调试很有用: 你能确认记忆是否真的写进去了.
        evergreen = self.load_evergreen()
        daily_files = list(self.memory_dir.glob("*.jsonl")) if self.memory_dir.is_dir() else []
        total_entries = 0
        for f in daily_files:
            try:
                total_entries += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                pass
        return {"evergreen_chars": len(evergreen), "daily_files": len(daily_files), "daily_entries": total_entries}


# ---------------------------------------------------------------------------
# 记忆工具: memory_write + memory_search
# ---------------------------------------------------------------------------

memory_store = MemoryStore(WORKSPACE_DIR)


def tool_memory_write(content: str, category: str = "general") -> str:
    # 这是给模型调用的 Python handler.
    # 模型不会直接执行 write_memory(), 它只会请求调用 memory_write 工具;
    # agent loop 收到 tool_use 后, 才会来到这里执行真实写入.
    print_tool("memory_write", f"[{category}] {content[:60]}...")
    return memory_store.write_memory(content, category)


def tool_memory_search(query: str, top_k: int = 5) -> str:
    # 工具返回值必须是字符串, 因为它会作为 tool_result 再喂回模型.
    # 这里保留 path + score + snippet, 让模型既知道内容, 也知道来源和可信度线索.
    print_tool("memory_search", query)
    results = memory_store.hybrid_search(query, top_k)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)


# ---------------------------------------------------------------------------
# 工具定义: Schema + Handler
# ---------------------------------------------------------------------------
# 工具 schema 设计说明:
#
# 每个章节 (s02, s06 等) 为了教学清晰度定义了自己的工具集.
# 在生产环境中, 工具 schema 会从共享注册表继承/组合.
#
# s06 中的工具 (memory_write, memory_search) 是对 s02 工具
# (bash, read_file, write_file, edit_file) 的补充 -- 而非替代.
# 完整的 agent 会将两组工具合并为一个列表传递给 LLM.

TOOLS = [
    {
        "name": "memory_write",
        # description 是模型决定"什么时候调用这个工具"的重要依据.
        # 写得太宽会导致模型乱记, 写得太窄会导致该记的时候不记.
        "description": (
            "Save an important fact or observation to long-term memory. "
            "Use when you learn something worth remembering about the user or context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                # content 是真正写入 JSONL 的文本.
                "content": {"type": "string", "description": "The fact or observation to remember."},
                # category 用于后续筛选和展示. 例如 preference/fact/context.
                "category": {"type": "string", "description": "Category: preference, fact, context, etc."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories for relevant information, ranked by similarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                # top_k 交给模型控制, 但 handler 有默认值.
                # 生产系统通常还会做范围限制, 例如 1 <= top_k <= 10.
                "top_k": {"type": "integer", "description": "Max results. Default: 5."},
            },
            "required": ["query"],
        },
    },
]

# TOOL_HANDLERS 是 schema 名称到 Python 函数的映射.
# 亮点:
#   schema 给模型看, handler 给程序执行.
#   这层映射让你可以审计/限制模型能触发的真实能力.
TOOL_HANDLERS: dict[str, Any] = {
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    # 所有工具调用统一从这里进入, 便于做错误处理、日志、权限控制.
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        # 模型请求了不存在的工具时, 不要崩溃; 把错误作为 tool_result 返回给模型.
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        # 参数 schema 和真实函数签名不一致时, 这里能给出清晰错误.
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        # 工具内部异常也转成字符串返回. 真实系统会额外记录 traceback.
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# 5. 系统提示词组装 -- 核心函数
# ---------------------------------------------------------------------------
# 教学演示 8 个关键提示词层级.
# 每轮重建 -- 上一轮可能更新了记忆.
# 模式: full (主 agent) / minimal (子 agent / cron) / none (最小化)


def build_system_prompt(
    mode: str = "full",
    bootstrap: dict[str, str] | None = None,
    skills_block: str = "",
    memory_context: str = "",
    agent_id: str = "main",
    channel: str = "terminal",
) -> str:
    """
    组装系统提示词.

    这是本文件的核心函数. 你可以把它理解成:
        静态文件 + 动态检索 + 运行时变量 -> 本轮 system prompt

    重要观念:
      - system prompt 不是启动时生成一次就永远不变.
      - 每轮用户消息进来后, 记忆召回结果可能不同, 当前时间也不同.
      - 所以 agent loop 会每轮重新调用这个函数.
    """
    if bootstrap is None:
        bootstrap = {}
    sections: list[str] = []

    # 第 1 层: 身份 -- 来自 IDENTITY.md 或默认值
    # 身份层放在最前面, 因为它定义"你是谁".
    # 如果连身份都没有, 就退回一个普通助手.
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful personal AI assistant.")

    # 第 2 层: 灵魂 -- 人格注入, 越靠前影响力越强
    # 亮点:
    #   SOUL.md 不是事实记忆, 而是行为风格和价值取向.
    #   只在 full 模式加入, 因为子任务/后台任务可能不需要完整人格.
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # 第 3 层: 工具使用指南
    # TOOLS.md 是"如何使用工具"的文字规范, 和 API 的 TOOLS schema 不同.
    # 例如它可以写: 什么时候该调用工具、失败后怎么处理、禁止做什么.
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # 第 4 层: 技能
    # 技能块来自 SkillsManager.format_prompt_block().
    # 它让模型知道有哪些专业工作流可以遵循, 但不直接授予执行权限.
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # 第 5 层: 记忆 -- 长期记忆 + 本轮自动搜索结果
    if mode == "full":
        mem_md = bootstrap.get("MEMORY.md", "").strip()
        parts: list[str] = []
        if mem_md:
            # Evergreen Memory 是长期稳定事实, 每轮 full prompt 都会带上.
            parts.append(f"### Evergreen Memory\n\n{mem_md}")
        if memory_context:
            # Recalled Memories 是针对本轮用户输入动态搜索出来的.
            # 同一个 agent, 不同用户问题, 这里注入的内容可能完全不同.
            parts.append(f"### Recalled Memories (auto-searched)\n\n{memory_context}")
        if parts:
            sections.append("## Memory\n\n" + "\n\n".join(parts))
        # 即使本轮没有记忆内容, 也要告诉模型"可以写记忆/搜记忆".
        # 否则模型可能不知道 memory_write 和 memory_search 的使用意图.
        sections.append(
            "## Memory Instructions\n\n"
            "- Use memory_write to save important user facts and preferences.\n"
            "- Reference remembered facts naturally in conversation.\n"
            "- Use memory_search to recall specific past information."
        )

    # 第 6 层: Bootstrap 上下文 -- 剩余的 Bootstrap 文件
    # 这些文件的语义更偏"运行规则/用户资料/项目约束", 所以放在身份、工具、
    # 技能、记忆之后. 它们依然重要, 但不应盖过 agent 的核心身份.
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # 第 7 层: 运行时上下文
    # 运行时上下文不是从磁盘来的, 而是本轮调用时才知道:
    # 当前 agent_id、模型、通道、时间、prompt 模式.
    # 这些信息能帮助模型调整输出, 也方便调试 /prompt.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"## Runtime Context\n\n"
        f"- Agent ID: {agent_id}\n- Model: {MODEL_ID}\n"
        f"- Channel: {channel}\n- Current time: {now}\n- Prompt mode: {mode}"
    )

    # 第 8 层: 渠道提示
    # 不同 channel 有不同输出约束:
    #   terminal 可以输出较完整 Markdown;
    #   Telegram 更适合短消息;
    #   Discord 有长度限制;
    #   Slack 有自己的 mrkdwn.
    # 这就是为什么 channel 也要进入 prompt.
    hints = {
        "terminal": "You are responding via a terminal REPL. Markdown is supported.",
        "telegram": "You are responding via Telegram. Keep messages concise.",
        "discord": "You are responding via Discord. Keep messages under 2000 characters.",
        "slack": "You are responding via Slack. Use Slack mrkdwn formatting.",
    }
    sections.append(f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}")

    # 用空行拼接每一层, 让最终 prompt 可读、可在 /prompt 中检查.
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 6. Agent 循环 + REPL
# ---------------------------------------------------------------------------

def handle_repl_command(
    cmd: str,
    bootstrap_data: dict[str, str],
    skills_mgr: SkillsManager,
    skills_block: str,
) -> bool:
    """
    处理 REPL 斜杠命令. 返回 True 表示已处理.

    这些命令不发给 LLM, 而是本地调试入口:
      /prompt    看最终系统提示词长什么样.
      /bootstrap 看启动时加载了哪些文件.
      /skills    看技能发现是否成功.
      /memory    看记忆统计.
      /search    手动测试记忆检索效果.
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/soul":
        # 直接展示人格文件, 方便你确认 SOUL.md 是否被加载.
        print_section("SOUL.md")
        soul = bootstrap_data.get("SOUL.md", "")
        print(soul if soul else f"{DIM}(未找到 SOUL.md){RESET}")
        return True

    if command == "/skills":
        # 展示技能元信息和路径. 如果某个技能没出现, 优先检查:
        #   1. 目录里是否有 SKILL.md
        #   2. frontmatter 里是否有 name
        #   3. 是否超过 MAX_SKILLS / MAX_SKILLS_PROMPT
        print_section("已发现的技能")
        if not skills_mgr.skills:
            print(f"{DIM}(未找到技能){RESET}")
        else:
            for s in skills_mgr.skills:
                print(f"  {BLUE}{s['invocation']}{RESET}  {s['name']} - {s['description']}")
                print(f"    {DIM}path: {s['path']}{RESET}")
        return True

    if command == "/memory":
        # 这是观察记忆系统是否写入成功的最轻量方式.
        print_section("记忆统计")
        stats = memory_store.get_stats()
        print(f"  长期记忆 (MEMORY.md): {stats['evergreen_chars']} 字符")
        print(f"  每日文件: {stats['daily_files']}")
        print(f"  每日条目: {stats['daily_entries']}")
        return True

    if command == "/search":
        # 手动搜索不会调用 LLM, 只走 MemoryStore.hybrid_search().
        # 如果自动召回效果不好, 先用这个命令调试检索质量.
        if not arg:
            print(f"{YELLOW}用法: /search <query>{RESET}")
            return True
        print_section(f"记忆搜索: {arg}")
        results = memory_store.hybrid_search(arg)
        if not results:
            print(f"{DIM}(无结果){RESET}")
        else:
            for r in results:
                color = GREEN if r["score"] > 0.3 else DIM
                print(f"  {color}[{r['score']:.4f}]{RESET} {r['path']}")
                print(f"    {r['snippet']}")
        return True

    if command == "/prompt":
        # /prompt 是学习本节最重要的命令:
        # 它让你看到 build_system_prompt() 的最终产物, 不再把系统提示词想成黑盒.
        print_section("完整系统提示词")
        prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=_auto_recall("show prompt"),
        )
        if len(prompt) > 3000:
            # 终端里不直接刷完整超长 prompt, 但会显示总长度.
            print(prompt[:3000])
            print(f"\n{DIM}... ({len(prompt) - 3000} more chars, total {len(prompt)}){RESET}")
        else:
            print(prompt)
        print(f"\n{DIM}提示词总长度: {len(prompt)} 字符{RESET}")
        return True

    if command == "/bootstrap":
        # 展示每个 Bootstrap 文件加载后的字符数.
        # 如果某个文件特别大, 这里能帮助你发现它是否被截断.
        print_section("Bootstrap 文件")
        if not bootstrap_data:
            print(f"{DIM}(未加载 Bootstrap 文件){RESET}")
        else:
            for name, content in bootstrap_data.items():
                print(f"  {BLUE}{name}{RESET}: {len(content)} chars")
        total = sum(len(v) for v in bootstrap_data.values())
        print(f"\n  {DIM}总计: {total} 字符 (上限: {MAX_TOTAL_CHARS}){RESET}")
        return True

    return False


def _auto_recall(user_message: str) -> str:
    """根据用户消息自动搜索相关记忆, 注入到系统提示词中."""
    # 亮点:
    #   用户没有输入 "/search" 时, agent 仍然会自动搜索记忆.
    #   这就是"记忆像本能一样参与每轮对话", 而不是一个需要用户显式触发的功能.
    results = memory_store.hybrid_search(user_message, top_k=3)
    if not results:
        return ""
    # 这里返回 Markdown bullet, 因为它最终会嵌入系统提示词的 Memory 层.
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)


def agent_loop() -> None:
    # 启动阶段: 加载 Bootstrap 文件, 发现技能 (技能仅在启动时发现一次)
    # 为什么技能只发现一次:
    #   技能目录扫描相对稳定, 每轮扫描会浪费时间.
    #   如果你新增了 SKILL.md, 重启这个脚本即可重新发现.
    loader = BootstrapLoader(WORKSPACE_DIR)
    bootstrap_data = loader.load_all(mode="full")

    skills_mgr = SkillsManager(WORKSPACE_DIR)
    skills_mgr.discover()
    skills_block = skills_mgr.format_prompt_block()

    # messages 保存对话历史, 格式遵循 Anthropic Messages API.
    # 注意: system prompt 不放在 messages 里, 而是每次 API 调用单独传 system=...
    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 06: Intelligence")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workspace: {WORKSPACE_DIR}")
    print_info(f"  Bootstrap 文件: {len(bootstrap_data)}")
    print_info(f"  已发现技能: {len(skills_mgr.skills)}")
    stats = memory_store.get_stats()
    print_info(f"  记忆: 长期 {stats['evergreen_chars']}字符, {stats['daily_files']} 个每日文件")
    print_info("  命令: /soul /skills /memory /search /prompt /bootstrap")
    print_info("  输入 'quit' 或 'exit' 退出.")
    print_info("=" * 60)
    print()

    while True:
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

        # REPL 命令在本地处理, 不进入 LLM 对话历史.
        # 这避免 /prompt /memory 这类调试命令污染正常聊天上下文.
        if user_input.startswith("/"):
            if handle_repl_command(user_input, bootstrap_data, skills_mgr, skills_block):
                continue

        # 自动记忆搜索 -- 将相关记忆注入系统提示词
        memory_context = _auto_recall(user_input)
        if memory_context:
            print_info("  [自动召回] 找到相关记忆")

        # 每轮重建系统提示词 (记忆可能在上一轮被更新)
        # 这是本节的关键闭环:
        #   上一轮模型可能调用 memory_write 写入新事实.
        #   本轮用户输入进来后, _auto_recall() 可能把这条事实搜出来.
        #   build_system_prompt() 再把它放进 Memory 层.
        system_prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=memory_context,
        )

        # 用户消息进入对话历史. 后续如果模型调用工具,
        # assistant tool_use 和 user tool_result 也会继续 append 到 messages.
        messages.append({"role": "user", "content": user_input})

        # Agent 内循环: 处理连续的工具调用直到 end_turn
        # 一轮用户输入可能触发多次工具调用:
        #   LLM -> tool_use(memory_search)
        #   程序执行工具 -> tool_result
        #   LLM 看到结果 -> 可能继续 tool_use(memory_write)
        #   最后 end_turn 输出自然语言答案
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID, max_tokens=8096,
                    system=system_prompt, tools=TOOLS, messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                # API 失败时回滚本轮消息, 避免半截失败对话留在历史里.
                # 这样用户修好网络/API key 后可以重新输入同一问题.
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # end_turn 表示模型已经完成本轮回答, 没有继续请求工具.
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if text:
                    print_assistant(text)
                break
            elif response.stop_reason == "tool_use":
                # tool_use 表示模型没有直接回答, 而是请求程序帮它执行工具.
                # 这里逐个执行 tool_use block, 并把结果按 tool_result 格式回传.
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                # Anthropic API 约定: tool_result 作为下一条 user 消息发回模型.
                messages.append({"role": "user", "content": tool_results})
                continue
            else:
                # 其它 stop_reason 例如 max_tokens, stop_sequence 等.
                # 教学版直接打印已有文本, 方便观察发生了什么.
                print_info(f"[stop_reason={response.stop_reason}]")
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if text:
                    print_assistant(text)
                break


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    # main() 只做启动前检查, 保持 agent_loop() 专注对话流程.
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}错误: 未设置 ANTHROPIC_API_KEY.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的密钥.{RESET}")
        sys.exit(1)
    if not WORKSPACE_DIR.is_dir():
        print(f"{YELLOW}错误: 未找到工作区目录: {WORKSPACE_DIR}{RESET}")
        print(f"{DIM}请从 claw0 项目根目录运行.{RESET}")
        sys.exit(1)
    agent_loop()


if __name__ == "__main__":
    main()
