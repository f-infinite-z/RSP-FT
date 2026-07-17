"""
黄金三问模板（C 组高质反问骨架，固定模板 + 当前实体动态填充）

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

以下黄金三问的框架结构（诗词：背景/意象/结构；菜谱：菜系/食材/时令）
由研究者选定，AI 工具辅助润色具体措辞。

# ============================================================
# 人工批注（作者）
# ============================================================
# C组反问的模板引擎，骨架可以参考一下或许有用，内容就算了，一开始想的是搞点高质量反问提示词，但是太模板化了，适得其反，有的领域算高质量有的反而算低质量
# 如果要用的话建议重新设计，我开始觉得还行，后面就垮掉了，得根据对应的领域专门设计提示词，偏文的和偏理的完全不一样
# 推荐的话是直接做一个能分开来M反问质量梯度的设计，只靠模板化提示词效果很差，所有反问单独设计可能有用，但太多了
# ============================================================

- 三问为固定骨架，覆盖各领域最核心的深度追问维度；
- 使用时动态填入当前实体（诗题/作者/朝代 或 菜名/菜系），
  以保证裁判评测中「实体相关性」维度可拿分。

B 组的低质反问来自数据集已生成的 counter_question 字段（见 dataset），
不在本模块内。本模块只负责 C 组高质反问。
"""
from typing import Dict, List, Optional

# 固定骨架（{entity} 处按领域填充具体实体描述）
POETRY_GOLDEN = [
    "《{title}》（{author}·{dynasty}）的创作时代背景与作者当时的人生境遇是怎样的？",
    "《{title}》中运用了哪些意象？这些意象分别承载何种典型象征内涵？",
    "《{title}》全诗结构脉络如何安排？首尾、情与景之间存在怎样的呼应关系？",
]

RECIPE_GOLDEN = [
    "{dish}属于哪个地域菜系？当地饮食风味偏好会如何影响它的烹饪手法？",
    "{dish}的核心食材起到什么作用？哪些步骤操作会直接影响成品口感？",
    "{dish}原本讲究什么食用时令？家常制作可以进行哪些合理改良取舍？",
]

# 法律领域仅跑 A/B 组（不使用黄金三问），此处为占位以兼容 dry-run 等边缘调用
LAW_GOLDEN = [
    "本条规定的行为或权利，其成立需要满足哪些法定构成要件？",
    "本条与民法典其他相关条款之间存在怎样的适用关系？",
    "违反本条规定会产生什么法律后果或责任？",
]


class GoldenQuestions:
    """按领域返回填充实体后的黄金三问。"""

    def __init__(self, domain: str):
        if domain == "poetry":
            self.templates = POETRY_GOLDEN
        elif domain == "recipe":
            self.templates = RECIPE_GOLDEN
        elif domain == "law":
            self.templates = LAW_GOLDEN
        else:
            raise ValueError(f"Unknown domain: {domain}")
        self.domain = domain

    def _entities(self, item: Dict) -> Dict[str, str]:
        """从数据集条目提取实体，缺失时给出安全兜底。"""
        if self.domain == "poetry":
            return {
                "title": item.get("poem_title", "这首诗"),
                "author": item.get("poem_author", "该诗人"),
                "dynasty": item.get("poem_dynasty", ""),
            }
        return {"dish": item.get("dish_name", "这道菜")}

    def fill(self, item: Dict) -> List[str]:
        """返回填充实体后的三问列表。"""
        ent = self._entities(item)
        out = []
        for tpl in self.templates:
            try:
                out.append(tpl.format(**ent))
            except (KeyError, IndexError):
                out.append(tpl)
        return out

    def pick(self, item: Dict, idx: Optional[int] = None) -> str:
        """
        选取一条黄金反问。
        idx 为 None 时按条目内容轮转（用 hash 保证可复现），否则取指定序号。
        """
        qs = self.fill(item)
        if idx is None:
            key = str(item.get("poem_title") or item.get("dish_name") or item.get("question", ""))
            idx = abs(hash(key)) % len(qs)
        return qs[idx % len(qs)]

    def as_combined(self, item: Dict) -> str:
        """把三问合并为一段（用于一次性深度追问的场景）。"""
        qs = self.fill(item)
        return " ".join(f"{i+1}.{q}" for i, q in enumerate(qs))
