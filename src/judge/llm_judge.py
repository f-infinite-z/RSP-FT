"""
第三方大模型裁判模块（双套四维评分 + 专长加权聚合 + 正向评分规范）

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

以下评分 Prompt 的四维维度与分数锚点由作者设计，AI 工具辅助润色措辞。

# ============================================================
# 人工批注（作者）
# ============================================================
# 项目设计阶段考虑到人数、资金等因素，直接采用了第三方大模型来打分，有条件的话可以引入人来评价的环节，我只做了适度的人工批注校验了几个小领域，
# 人和第三方大模型的打分结果都是正相关的，基本上没什么问题。可以尝试用更多裁判模型来保障偏向客观，单模型的话不推荐，单第三方模型有很强的偏好性
# 然后反问质量和回答质量的几个维度、比例以及最终评分划分，我这边仅供参考，最好是用更贴近你想要尝试的领域，采用更合适的维度，以及比例
# 经验分享：主流的大模型我基本上都尝试过，我这边选的三个是我觉得最合适我的，项目训练是租用的国内云GPU，所以用国模会更稳定一些，免得训练过程中环境波动
# 重跑损失不小，根据我的使用心得，基本上各个模型的风格和其公司的状态是高度相关性的，猜测是与训练时的数据源相关，天然的多选取已有数据，以及对应人工标注的偏好
# 所以我这边选了逻辑性强的ds，中文领域偏文学性的豆包，以及做中立的千问。如果想复现的话可以尝试自己常用的或者自己感兴趣的模型。下面相关的模型名字记得改，不然跑不起来
# ============================================================

  - 回答质量 Y（主指标，因变量）：事实准确性(权重最高)/内容完备度/逻辑结构性/信息增益深度
  - 反问质量 M（中介变量）      ：实体相关性/信息增益/可行性/递进性

三裁判最终方案（DeepSeek-V4-Pro / Doubao-Seed-Lite / Qwen3.7-Plus）：
  - 低成本模型 + 强约束弥补（Promp梯度要求 + temperature=0.05 + top_p=0.2）
  - 均对全部维度打分，保留 Kendall's W 一致性检验
  - 聚合时按各家专长在不同维度上加权（方案B）：
      Doubao-Seed-Lite: 文学情感、赏析深度、语言流畅
      DeepSeek-V4-Pro : 逻辑结构、事实正误、标准化打分
      Qwen3.7-Plus    : 垂直领域知识、中立尺度、菜谱/文史事实复核
"""
import json
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

DOMAIN_DESC = {"poetry": "古诗词鉴赏与评论", "recipe": "中华菜谱与烹饪知识"}

# 评分规范（注入所有裁判 System Prompt，正向引导）
# 默认使用严格模式，若小数据验证发现中段缺失/两极挤压，切为宽松模式
SCORING_CONSTRAINT = (
    "【评分正向规范】"
    "1. 完整使用 1–10 全区间灵活打分，充分区分文本细微优劣："
    "1–2 极差、3–4 不足、5–6 合格、7–8 良好、9–10 优秀；"
    "同分段内需进一步细分梯度，例如 7 分为偏弱良好、8 分为偏优良好，"
    "不要扎堆集中在同一档分数。"
    "2. 仅浅层复述、无拓展分析的内容，深度/完备度维度最高不超过 6 分；"
    "存在关键事实错误时，事实准确性维度最高不超过 3 分。"
)

SCORING_CONSTRAINT_SOFT = (
    "【评分正向规范】"
    "1. 请在 1–10 区间内灵活分配分数，依据文本真实质量客观区分优劣："
    "1–2 极差、3–4 不足、5–6 合格、7–8 良好、9–10 优秀；"
    "同分段内可根据细微差异拉开区分度，允许各分数段均匀出现。"
)

# 单维度内的加权（用于 overall 计算）：事实准确性权重最高
ANSWER_WEIGHTS = {"factual": 0.4, "completeness": 0.2, "logic": 0.2, "depth": 0.2}
CQ_WEIGHTS = {"relevance": 0.25, "gain": 0.25, "feasibility": 0.25, "progression": 0.25}

# 裁判专长身份（写入 system prompt；保留用于消融分析）
JUDGE_PERSONA = {
    "doubao": "你尤其擅长文学情感把握、赏析深度与语言表达的评判。",
    "deepseek": "你尤其擅长逻辑结构分析、事实正误判断与标准化评分。",
    "qwen": "你尤其擅长垂直领域知识校准、中立尺度平衡与文史/菜谱事实复核。",
}

# 专长加权矩阵：每个维度上三裁判的权重（同维度三家之和=1）
# 行=维度，列=裁判。缺失裁判时按现有裁判权重归一化。
ANSWER_JUDGE_WEIGHTS = {
    #                 doubao  deepseek  qwen
    "factual":       {"doubao": 0.20, "deepseek": 0.40, "qwen": 0.40},
    "completeness":  {"doubao": 0.30, "deepseek": 0.30, "qwen": 0.40},
    "logic":         {"doubao": 0.20, "deepseek": 0.50, "qwen": 0.30},
    "depth":         {"doubao": 0.50, "deepseek": 0.25, "qwen": 0.25},
}
CQ_JUDGE_WEIGHTS = {
    "relevance":     {"doubao": 0.30, "deepseek": 0.30, "qwen": 0.40},
    "gain":          {"doubao": 0.40, "deepseek": 0.30, "qwen": 0.30},
    "feasibility":   {"doubao": 0.25, "deepseek": 0.50, "qwen": 0.25},
    "progression":   {"doubao": 0.50, "deepseek": 0.25, "qwen": 0.25},
}


@dataclass
class AnswerScore:
    """回答质量 Y。"""
    factual: float          # 事实准确性（权重最高）
    completeness: float     # 内容完备度
    logic: float            # 逻辑结构性与条理性
    depth: float            # 信息增益 / 分析深度

    @property
    def overall(self) -> float:
        return (self.factual * ANSWER_WEIGHTS["factual"]
                + self.completeness * ANSWER_WEIGHTS["completeness"]
                + self.logic * ANSWER_WEIGHTS["logic"]
                + self.depth * ANSWER_WEIGHTS["depth"])

    def to_dict(self) -> dict:
        return {"factual": self.factual, "completeness": self.completeness,
                "logic": self.logic, "depth": self.depth, "overall": round(self.overall, 3)}


@dataclass
class CQScore:
    """反问质量 M。"""
    relevance: float        # 实体相关性
    gain: float             # 信息增益
    feasibility: float      # 可行性
    progression: float      # 递进性

    @property
    def overall(self) -> float:
        return (self.relevance + self.gain + self.feasibility + self.progression) / 4

    def to_dict(self) -> dict:
        return {"relevance": self.relevance, "gain": self.gain,
                "feasibility": self.feasibility, "progression": self.progression,
                "overall": round(self.overall, 3)}


# ---- Prompt 构造 -----------------------------------------------------------
# 以下评分 Prompt 的四维维度定义、分数锚点由研究者设计，
# AI 工具辅助润色了措辞与格式。----

def build_answer_judge_prompt(domain: str, question: str, answer: str) -> str:
    dom = DOMAIN_DESC.get(domain, domain)
    return f"""你是一位{dom}领域专家评审。请对以下回答在 4 个维度上进行 1-10 分评分（事实准确性权重最高，遵循评分规范）：

问题：{question}
回答：{answer}

评分维度：
1. 事实准确性(factual)：是否符合客观史实/文本原文/传统知识，无编造曲解。
   → 9-10=全部有据可查无偏差；7-8=偶有次要偏差不损核心；5-6=局部缺陷但主体可靠；3-4=多处错误动摇结论；1-2=大量虚构关键事实全错
2. 内容完备度(completeness)：是否覆盖用户诉求核心要点。
   → 9-10=完整覆盖兼顾多层次；7-8=要点齐全个别次要角度不足；5-6=基础要点具备但重要角度缺失；3-4=严重残缺仅触及皮毛；1-2=答非所问
3. 逻辑结构性(logic)：逻辑连贯、层次清晰、前后自洽。
   → 9-10=层次分明递进严谨；7-8=条理清晰偶有跳跃；5-6=基本通顺缺框架；3-4=松散混乱难跟逻辑；1-2=碎片化无结构
4. 信息增益/分析深度(depth)：是否超越表层复述进行推导解读，是否体现{dom}专业细节。
   → 9-10=体系化深度分析展现专业洞见；7-8=有分析延伸且有独立见解；5-6=有初步延伸但停留表面；3-4=仅陈述基础无延伸；1-2=空洞重复常识

请输出 JSON（仅JSON）：{{"factual": X, "completeness": X, "logic": X, "depth": X}}"""


def build_cq_judge_prompt(domain: str, question: str, counter_question: str) -> str:
    dom = DOMAIN_DESC.get(domain, domain)
    return f"""你是一位{dom}领域评测裁判。给定用户原始提问与模型提出的反问，评估该反问是否贴合原文主题、是否有助于产出高质量回答。请在 4 个维度上 1-10 分打分（遵循评分规范）：

原始问题：{question}
模型反问：{counter_question}

评分维度：
1. 实体相关性(relevance)：反问是否紧扣原文/原菜的具体实体，而非泛泛而问。
   → 9-10=精确指向具体意象/食材/技法；7-8=主体相关偶有泛化；5-6=领域相关但泛化；3-4=勉强沾边；1-2=完全脱离原文
2. 信息增益(gain)：反问能否引出原答案未覆盖的新信息。
   → 9-10=导向更深层次知识盲区；7-8=有明显增量突破表层；5-6=有边际补充但有限；3-4=几无信息增量；1-2=重复已知无增益
3. 可行性(feasibility)：反问是否清晰、可被有效回答。
   → 9-10=命题明确可操作性极强；7-8=清晰且基本可答；5-6=方向清晰但不够精确；3-4=含混难回答；1-2=无法作答
4. 递进性(progression)：反问是否推动理解走向更深层次。
   → 9-10=层层递进从现象到本质；7-8=有明显推进进入更深层；5-6=有追问但停留同层；3-4=原地打转无推进；1-2=倒退/偏离主题

请输出 JSON（仅JSON）：{{"relevance": X, "gain": X, "feasibility": X, "progression": X}}"""


# ---- 解析 ------------------------------------------------------------------

def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return raw.strip()


def parse_answer_score(raw: str) -> Optional[AnswerScore]:
    try:
        d = json.loads(_clean_json(raw))
        return AnswerScore(float(d["factual"]), float(d["completeness"]),
                           float(d["logic"]), float(d["depth"]))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def parse_cq_score(raw: str) -> Optional[CQScore]:
    try:
        d = json.loads(_clean_json(raw))
        return CQScore(float(d["relevance"]), float(d["gain"]),
                       float(d["feasibility"]), float(d["progression"]))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


# ---- 对比评分 Prompt（一次调用比较 A/B/C 三组，含 CoT 推理）----------

def build_compare_answer_prompt(domain: str, question: str, answer_a: str, answer_b: str, answer_c: str) -> str:
    dom = DOMAIN_DESC.get(domain, domain)
    inner_a = '{"factual":X,"completeness":X,"logic":X,"depth":X}'
    json_tmpl = '{"A":' + inner_a + ',"B":' + inner_a + ',"C":' + inner_a + '}'
    return f"""你是一位{dom}领域专家评审。以下是对同一个问题的三份匿名回答（分别标记为 A、B、C），请按 1-10 分评分（遵循评分规范）。

问题：{question}

===== 回答 A =====
{answer_a}
===== 回答 B =====
{answer_b}
===== 回答 C =====
{answer_c}

评分维度（每个维度 1-10 分）：
1. factual（事实准确性）：9-10=全部有据可查无偏差；7-8=偶有次要偏差不损核心；5-6=局部缺陷但主体可靠；3-4=多处错误动摇结论；1-2=大量虚构关键事实全错
2. completeness（内容完备度）：9-10=完整覆盖兼顾多层次；7-8=要点齐全个别次要角度不足；5-6=基础要点具备但重要角度缺失；3-4=严重残缺仅触及皮毛；1-2=答非所问
3. logic（逻辑结构性）：9-10=层次分明递进严谨；7-8=条理清晰偶有跳跃；5-6=基本通顺缺框架；3-4=松散混乱难跟逻辑；1-2=碎片化无结构
4. depth（信息增益/分析深度）：9-10=体系化深度分析展现专业洞见；7-8=有分析延伸且有独立见解；5-6=有初步延伸但停留表面；3-4=仅陈述基础无延伸；1-2=空洞重复常识

输出格式（先写推理，后出 JSON）：
reasoning: 2-3 句话比较三份回答的主要差异与优劣判断
json: {json_tmpl}"""


def build_compare_cq_prompt(domain: str, question: str, cq_b: str, cq_c: str) -> str:
    dom = DOMAIN_DESC.get(domain, domain)
    inner = '{"relevance":X,"gain":X,"feasibility":X,"progression":X}'
    json_tmpl = '{"B":' + inner + ',"C":' + inner + '}'
    return f"""你是一位{dom}领域评测裁判。以下是对同一个问题的两份匿名反问（分别标记为 B、C），请按 1-10 分评分（遵循评分规范）。

原始问题：{question}

===== 反问 B =====
{cq_b}
===== 反问 C =====
{cq_c}

评分维度（每个维度 1-10 分）：
1. relevance（实体相关性）：9-10=精确指向具体意象/食材/技法；7-8=主体相关偶有泛化；5-6=领域相关但泛化；3-4=勉强沾边；1-2=完全脱离原文
2. gain（信息增益）：9-10=导向更深层次知识盲区；7-8=有明显增量突破表层；5-6=有边际补充但有限；3-4=几无信息增量；1-2=重复已知无增益
3. feasibility（可行性）：9-10=命题明确可操作性极强；7-8=清晰且基本可答；5-6=方向清晰但不够精确；3-4=含混难回答；1-2=无法作答
4. progression（递进性）：9-10=层层递进从现象到本质；7-8=有明显推进进入更深层；5-6=有追问但停留同层；3-4=原地打转无推进；1-2=倒退/偏离主题

输出格式（先写推理，后出 JSON）：
reasoning: 2-3 句话比较两份反问的主要差异与质量判断
json: {json_tmpl}"""


# ---- 对比评分解析 -----------------------------------------------------------

def parse_compare_answer(raw: str) -> Optional[Dict[str, AnswerScore]]:
    try:
        text = raw.strip()
        json_str = text
        if "json:" in text:
            json_str = text.split("json:", 1)[1].strip()
        d = json.loads(_clean_json(json_str))
        return {k: AnswerScore(float(v["factual"]), float(v["completeness"]),
                               float(v["logic"]), float(v["depth"]))
                for k, v in d.items()}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def parse_compare_cq(raw: str) -> Optional[Dict[str, CQScore]]:
    try:
        text = raw.strip()
        json_str = text
        if "json:" in text:
            json_str = text.split("json:", 1)[1].strip()
        d = json.loads(_clean_json(json_str))
        return {k: CQScore(float(v["relevance"]), float(v["gain"]),
                           float(v["feasibility"]), float(v["progression"]))
                for k, v in d.items()}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None

def cross_judge(scores_per_judge: Dict) -> Dict:
    """三裁判交叉评分（简单均值聚合，主结论用）。"""
    valid = {k: v for k, v in scores_per_judge.items() if v is not None}
    if len(valid) < 1:
        return {"mean_overall": None, "n_judges": 0}
    overalls = {name: s.overall for name, s in valid.items()}
    mean = round(float(np.mean(list(overalls.values()))), 3)
    std = round(float(np.std(list(overalls.values()), ddof=1)), 3) if len(overalls) >= 2 else 0.0
    max_diff = round(max(overalls.values()) - min(overalls.values()), 2) if len(overalls) >= 2 else 0.0
    return {
        "mean_overall": mean,
        "std": std,
        "max_diff": max_diff,
        "n_judges": len(valid),
        "per_judge": {name: s.to_dict() for name, s in valid.items()},
        "disagreement": max_diff > 2.0,  # 裁判间分差 > 2 标记分歧
    }


def _weighted_dim(scores_per_judge: Dict, dim: str, weight_matrix: Dict) -> Optional[float]:
    """对单个维度按专长权重聚合；缺失裁判则在现有裁判间归一化。"""
    w = weight_matrix[dim]
    num, den = 0.0, 0.0
    for name, s in scores_per_judge.items():
        if s is None or name not in w:
            continue
        val = getattr(s, dim)
        num += w[name] * val
        den += w[name]
    return (num / den) if den > 0 else None


def expertise_aggregate(scores_per_judge: Dict, kind: str) -> Dict:
    """
    专长加权聚合（仅用于补充消融分析，非主结论）。
    主结论请使用 cross_judge() 的简单均值聚合。
    kind: "answer" -> 回答质量 Y；"cq" -> 反问质量 M。
    """
    valid = {k: v for k, v in scores_per_judge.items() if v is not None}
    if not valid:
        return {"mean_overall": None, "n_judges": 0}

    if kind == "answer":
        jw, dw, dims = ANSWER_JUDGE_WEIGHTS, ANSWER_WEIGHTS, list(ANSWER_WEIGHTS)
    else:
        jw, dw, dims = CQ_JUDGE_WEIGHTS, CQ_WEIGHTS, list(CQ_WEIGHTS)

    dim_scores = {d: _weighted_dim(valid, d, jw) for d in dims}
    overall = sum(dim_scores[d] * dw[d] for d in dims if dim_scores[d] is not None)

    per_judge_overall = {name: s.overall for name, s in valid.items()}
    vals = list(per_judge_overall.values())
    max_diff = round(max(vals) - min(vals), 2) if len(vals) >= 2 else 0.0

    return {
        "mean_overall": round(overall, 3),
        "dim_weighted": {d: round(v, 3) for d, v in dim_scores.items() if v is not None},
        "max_diff": max_diff,
        "n_judges": len(valid),
        "per_judge": {name: s.to_dict() for name, s in valid.items()},
        "disagreement": max_diff > 2.0,
    }


def kendall_w(all_scores: List[List[float]]) -> Dict:
    """
    Kendall's W 一致性系数（含同分校正项），评估多裁判整体一致性。
    all_scores: [[法官1所有样本得分], [法官2所有样本得分], ...]  shape=(m, n)
    返回 W (0-1，越高越一致) 及基本统计。
    """
    scores = np.array(all_scores)
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 2:
        return {"W": 1.0, "n_items": 0, "n_judges": 0}

    m, n = scores.shape  # m=裁判数, n=样本数

    # 每个裁判对 n 个样本排名（降序，同分取平均排名）
    from scipy.stats import rankdata
    ranks = np.array([rankdata(-s) for s in scores])

    # 同分校正项 T = Σ(t³ - t)，t 为每组同分个数
    T = 0.0
    for i in range(m):
        _, counts = np.unique(ranks[i], return_counts=True)
        T += float(np.sum(counts**3 - counts))

    R_j = np.sum(ranks, axis=0)          # 每个样本的总排名
    S = np.sum((R_j - np.mean(R_j))**2)   # 离差平方和

    denom = m**2 * (n**3 - n) - m * T
    W = (12.0 * S) / denom if denom > 0 else 0.0

    return {"W": round(max(0.0, min(1.0, float(W))), 4),
            "n_items": n, "n_judges": m}


class Judge:
    """封装单个裁判客户端，评回答 Y 或反问 M。"""

    MAX_INPUT_TOKENS = 4000      # 单次评分输入上限（中文字符估算 / 1.3）
    MAX_INPUT_TOKENS_COMPARE = 6000  # 对比评分输入上限（含三份答案）
    MAX_RETRIES = 3
    CIRCUIT_BREAKER_THRESHOLD = 5   # 连续失败 >= 此数则熔断
    CIRCUIT_COOLDOWN_SEC = 60       # 熔断冷却时间

    def __init__(self, client, model: str, domain: str, name: str = "",
                 max_tokens: int = 300, max_tokens_compare: int = 1000,
                 scoring_mode: str = "strict"):
        self.client = client
        self.model = model
        self.domain = domain
        self.name = name
        self.persona = JUDGE_PERSONA.get(name, "")
        self.max_tokens = max_tokens
        self.max_tokens_compare = max_tokens_compare
        self.scoring_mode = scoring_mode
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = 0.0

    # ---- 令牌上限保护 ----------------------------------------------------------

    @staticmethod
    def _est_tokens(text: str) -> int:
        """粗略估算中文字符 token 数（≈ 字符数 / 1.3）。"""
        return max(1, int(len(text) / 1.3))

    def _truncate(self, text: str, max_tokens: int, label: str = "") -> str:
        """若超出上限则截断：保留前 60% + 后 30%（中间省略提示）。"""
        est = self._est_tokens(text)
        if est <= max_tokens:
            return text
        head_ratio = 0.6
        tail_ratio = 0.3
        head_len = int(len(text) * head_ratio)
        tail_len = int(len(text) * tail_ratio)
        truncated = text[:head_len] + "\n…[中间内容已截断，超出令牌上限]…\n" + text[-tail_len:]
        if label:
            print(f"  [token-guard] {label} {est}→~{self._est_tokens(truncated)} tokens")
        return truncated

    # ---- 重试 + 熔断 -----------------------------------------------------------

    def _safe_call(self, prompt: str, max_tokens: int = 300, json_mode: bool = True) -> Optional[str]:
        """带重试+熔断+指数退避的 API 调用。失败返回 None。"""
        # 熔断检查
        if self._circuit_open:
            if time.time() - self._last_failure_time > self.CIRCUIT_COOLDOWN_SEC:
                self._circuit_open = False
                self._failure_count = 0
                print(f"  [circuit] {self.name} 熔断冷却结束，重新尝试")
            else:
                print(f"  [circuit] {self.name} 熔断中，跳过调用（{self._failure_count}次连续失败）")
                return None

        messages = []
        system_parts = []
        if self.persona:
            system_parts.append(self.persona)
        constr = SCORING_CONSTRAINT_SOFT if self.scoring_mode == "soft" else SCORING_CONSTRAINT
        system_parts.append(constr)
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.model, "temperature": 0.05, "max_tokens": max_tokens,
                  "top_p": 0.2, "messages": messages,
                  "extra_body": {"thinking": {"type": "disabled"}}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._failure_count = 0  # 成功后清零
                return resp.choices[0].message.content
            except Exception as e:
                self._failure_count += 1
                wait = 2 ** attempt  # 指数退避：1s, 2s, 4s
                print(f"  [retry] {self.name} 第{attempt+1}次调用失败({e})，{wait}s后重试...")
                if self._failure_count >= self.CIRCUIT_BREAKER_THRESHOLD:
                    self._circuit_open = True
                    self._last_failure_time = time.time()
                    print(f"  [circuit] {self.name} 熔断触发（连续{self._failure_count}次失败），暂停{self.CIRCUIT_COOLDOWN_SEC}s")
                    return None
                time.sleep(wait)
        return None

    def _call(self, prompt: str) -> Optional[str]:
        return self._safe_call(prompt, max_tokens=self.max_tokens, json_mode=True)

    def _call_compare(self, prompt: str) -> Optional[str]:
        return self._safe_call(prompt, max_tokens=self.max_tokens_compare, json_mode=False)

    def score_answer(self, question: str, answer: str) -> Optional[AnswerScore]:
        answer = self._truncate(answer, self.MAX_INPUT_TOKENS, f"{self.name}:answer")
        raw = self._call(build_answer_judge_prompt(self.domain, question, answer))
        return parse_answer_score(raw) if raw else None

    def score_cq(self, question: str, counter_question: str) -> Optional[CQScore]:
        raw = self._call(build_cq_judge_prompt(self.domain, question, counter_question))
        return parse_cq_score(raw) if raw else None

    def compare_answer(self, question: str, answer_a: str, answer_b: str, answer_c: str) -> Optional[Dict[str, AnswerScore]]:
        """一次 API 调用比较 A/B/C 三份回答，含 CoT 推理。返回 {"A":AnswerScore, "B":..., "C":...}"""
        answer_a = self._truncate(answer_a, self.MAX_INPUT_TOKENS_COMPARE // 3, f"{self.name}:answer_a")
        answer_b = self._truncate(answer_b, self.MAX_INPUT_TOKENS_COMPARE // 3, f"{self.name}:answer_b")
        answer_c = self._truncate(answer_c, self.MAX_INPUT_TOKENS_COMPARE // 3, f"{self.name}:answer_c")
        raw = self._call_compare(build_compare_answer_prompt(
            self.domain, question, answer_a, answer_b, answer_c))
        return parse_compare_answer(raw) if raw else None

    def compare_cq(self, question: str, cq_b: str, cq_c: str) -> Optional[Dict[str, CQScore]]:
        """一次 API 调用比较 B/C 两份反问，含 CoT 推理。返回 {"B":CQScore, "C":CQScore}"""
        raw = self._call_compare(build_compare_cq_prompt(self.domain, question, cq_b, cq_c))
        return parse_compare_cq(raw) if raw else None
