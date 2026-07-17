"""
反问式自博弈——对话管理模块（A/B/C 三组分层操纵）

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

以下 Prompt 模板的方向、角色设定与关键约束由研究者设计，
AI 工具辅助润色措辞。

# ============================================================
# 人工批注（作者）
# ============================================================
# 这是本项目的核心设计，主要目前是将自然语言下的A/B/C,三组分组的对话流程逐步用代码实现然后编排起来构成完整结构
# 关键设计：B/C 流程完全对称，唯一系统差异=反问文本质量。这样不管B好还是C好，都只能归因于"反问质量"，而非"多了一轮对话"。
# 四个板块吧主要，定义变量，语法糖，提示词，整合对话
# ============================================================

三组构成反问强度 X 的分层操纵。
  A 组 (X=0) : 单向问答            Q -> 最终回答
  B 组 (X=1) : 往复 + 普通反问      Q -> 普通反问cq_low -> 仿真补充r -> 最终回答
  C 组 (X=2) : 往复 + 高质黄金反问   Q -> 高质反问cq_high -> 仿真补充r -> 最终回答

B、C 流程结构 100% 一致，唯一系统差异 = 反问质量（cq_low vs cq_high），
从而完全控制"有无往复对话"这一混淆变量，干净剥离反问质量的中介效应。

角色：
  M_A  提问者 + 回答反问（仿真用户补充），不更新梯度（见 counter_answer.py）
  M_B  回答者，被训练

本模块只负责"构造对话/样本"与"组织损失所需文本片段"，
不涉及梯度更新（由 self_play.py 调用）。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class Group(Enum):
    A = "A"   # 基线，单向，X=0
    B = "B"   # 对照，普通反问，X=1
    C = "C"   # 实验，黄金反问，X=2


X_LEVEL = {Group.A: 0, Group.B: 1, Group.C: 2}

"""语法糖"""
@dataclass
class DialogueSample:
    """一条对话产出的训练样本。"""
    group: str
    x_level: int
    question: str
    answer_first: str                          # 初答（B/C）或唯一回答（A）
    counter_question: Optional[str] = None     # 反问（A 组为 None）
    counter_source: Optional[str] = None       # 反问来源: none/normal/golden
    user_supplement: Optional[str] = None      # 仿真用户对反问的补充回答 r
    answer_final: Optional[str] = None         # 最终回答（B/C；A 组即 answer_first）
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "group": self.group, "x_level": self.x_level, "question": self.question,
            "answer_first": self.answer_first, "counter_question": self.counter_question,
            "counter_source": self.counter_source, "user_supplement": self.user_supplement,
            "answer_final": self.answer_final, "meta": self.meta,
        }

    @property
    def eval_answer(self) -> str:
        """裁判评判的最终回答（因变量 Y 的评测对象）。"""
        return self.answer_final if self.answer_final else self.answer_first


# ---- Prompt 构造 -------------------------------------------------------------
# 以下 System Prompt 模板：
#   方向（领域专家角色/反问机制/最终回答完善）和关键约束由研究者设计，
#   AI 工具辅助润色了措辞与表达格式。精简/严格模式的控制策略由研究者提出。----

import os

DOMAIN_NAME = {"recipe": "中华菜谱与烹饪", "poetry": "古诗词鉴赏", "law": "中国民法典"}

def _dom(domain: str) -> str:
    return DOMAIN_NAME.get(domain, domain)

ANSWER_SYS = "你是{domain}领域的专家，请准确、完整、有深度地回答用户的问题。"
CQ_SYS = ("你是{domain}领域的专家。在初步回答后，为了把问题理解得更透彻，"
          "请针对你自己的回答提出一个有价值的反问（追问关键细节、辨析或推理）。只输出反问本身。")
FINAL_SYS = ("你是{domain}领域的专家。请结合下面的追问与补充信息，"
             "给出一个比初次回答更准确、更深入的最终回答。")

"""长度控制（通过环境变量开关"""

# 精简模式：环境变量 CONCISE_MODE=1 开启，A/B/C 三组一致，用于"控制长度"快验证
_CONCISE = os.getenv("CONCISE_MODE", "") not in ("", "0", "false", "False")
_CONCISE_LIMIT = os.getenv("CONCISE_LIMIT", "150")
# 模式1(仅控长度)：只压缩篇幅，用于隔离"长度"单一变量
_CONCISE_SUFFIX = ("请先在心里梳理要点，再用简洁语言作答；"
                   f"只保留最关键的信息，控制在{_CONCISE_LIMIT}字以内，避免冗余重复。")
# 模式2(控长度+扣题防编造)：CONCISE_STRICT=1 开启，用于T1-plus
_STRICT = os.getenv("CONCISE_STRICT", "") not in ("", "0", "false", "False")
_STRICT_SUFFIX = ("请紧扣问题核心作答，只陈述你确定且与问题直接相关的信息；"
                  "不要为凑篇幅而编造、堆砌或偏离主题；宁可简短也不要包含不确定或无关内容，"
                  f"控制在{_CONCISE_LIMIT}字以内。")


"""对话编排 串起来，完整对话"""

def _maybe_concise(sys_text: str) -> str:
    if _STRICT:
        return sys_text + _STRICT_SUFFIX
    if _CONCISE:
        return sys_text + _CONCISE_SUFFIX
    return sys_text


def build_answer_prompt(domain: str, question: str) -> List[Dict]:
    return [
        {"role": "system", "content": _maybe_concise(ANSWER_SYS.format(domain=_dom(domain)))},
        {"role": "user", "content": question},
    ]


def build_cq_prompt(domain: str, question: str, answer_first: str) -> List[Dict]:
    return [
        {"role": "system", "content": CQ_SYS.format(domain=_dom(domain))},
        {"role": "user", "content": f"问题：{question}\n我的回答：{answer_first}\n请提出一个反问。"},
    ]


def build_final_prompt(domain: str, question: str, answer_first: str,
                       counter_question: str, user_supplement: str) -> List[Dict]:
    ctx = (f"原问题：{question}\n"
           f"我的初次回答：{answer_first}\n"
           f"我提出的反问：{counter_question}\n"
           f"针对反问的补充信息：{user_supplement}\n"
           f"请据此给出最终回答。")
    return [
        {"role": "system", "content": _maybe_concise(FINAL_SYS.format(domain=_dom(domain)))},
        {"role": "user", "content": ctx},
    ]


# ---- 对话编排 ---------------------------------------------------------------

GenerateFn = Callable[[List[Dict]], str]
AnswerCQFn = Callable[[str, str], str]      # (question, cq) -> 仿真用户补充 r
CQProvider = Callable[[], str]              # 返回反问文本（B: 普通 / C: 黄金）


def run_dialogue(group: Group,
                 domain: str,
                 question: str,
                 generate_fn: GenerateFn,
                 cq_provider: Optional[CQProvider] = None,
                 answer_cq_fn: Optional[AnswerCQFn] = None,
                 gold_answer: Optional[str] = None,
                 counter_source: str = "") -> DialogueSample:
    """
    执行单条对话，返回 DialogueSample。

    gold_answer 若提供（数据集标准答案），作为初答的教师信号，稳定 self-play 起步。
    cq_provider：B 组返回数据集普通反问，C 组返回黄金三问；A 组不使用。
    answer_cq_fn：M_A 依据反问生成仿真用户补充 r。
    """
    answer_first = gold_answer if gold_answer is not None else \
        generate_fn(build_answer_prompt(domain, question))

    if group == Group.A:
        return DialogueSample(
            group=group.value, x_level=X_LEVEL[group], question=question,
            answer_first=answer_first, counter_source="none",
            answer_final=answer_first)

    # B / C：生成反问
    if cq_provider is not None:
        counter_question = cq_provider()
    else:
        counter_question = generate_fn(build_cq_prompt(domain, question, answer_first))

    # M_A 回答反问（仿真用户补充）
    user_supplement = ""
    if answer_cq_fn is not None:
        user_supplement = answer_cq_fn(question, counter_question)

    answer_final = generate_fn(build_final_prompt(
        domain, question, answer_first, counter_question, user_supplement))

    return DialogueSample(
        group=group.value, x_level=X_LEVEL[group], question=question,
        answer_first=answer_first, counter_question=counter_question,
        counter_source=counter_source or ("golden" if group == Group.C else "normal"),
        user_supplement=user_supplement, answer_final=answer_final)
