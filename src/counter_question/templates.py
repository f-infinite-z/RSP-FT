"""
反问模板库
基于每个领域6种反问类型，提供结构化模板。

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

以下模板的类别划分（clarify/reasoning/meta）和每类的反问方向由研究者设计。

# ============================================================
# 人工批注（作者）
# ============================================================
# 反问设计参考库，拓展方向的话可以做更多层次的区分，这边项目主要是3层0%-50%-100%，渐进提升，可以做更多区分度尝试，但是注意，自由生成很慢，也可以不做到100%
# 下面反问库那些提示词可以根据你做实验的领域需求自己变换，这边以我自己项目为主，仅供参考
# ============================================================
"""

from enum import Enum
from typing import Dict, List


class CQType(Enum):
    CLARIFY = "clarify"
    REASONING = "reasoning"
    META = "meta"


class DomainCQTemplates:
    """领域反问模板管理器，支持模板→自由生成的渐进式策略"""

    def __init__(self, domain: str):
        if domain == "recipe":
            self.templates = RECIPE_TEMPLATES
        elif domain == "poetry":
            self.templates = POETRY_TEMPLATES
        else:
            raise ValueError(f"Unknown domain: {domain}")
        self.domain = domain

    def get_random(self, cq_type: CQType = None) -> Dict:
        """随机返回一个反问模板"""
        import random
        if cq_type:
            pool = [t for t in self.templates if t["type"] == cq_type.value]
        else:
            pool = self.templates
        return random.choice(pool) if pool else self.templates[0]

    def fill_template(self, template: Dict, **kwargs) -> str:
        """用具体值填充模板占位符"""
        text = template["template"]
        for key, value in kwargs.items():
            text = text.replace(f"[{key}]", str(value))
        return text

    def get_templates_for_stage(self, stage: int) -> List[Dict]:
        """
        渐进式反问策略：
        stage 1 (iter 1-2): 100% 模板
        stage 2 (iter 3-4): 50% 模板可用
        stage 3 (iter 5):   0% 模板（完全自由生成）
        """
        if stage == 1:
            return self.templates
        elif stage == 2:
            import random
            n = max(1, len(self.templates) // 2)
            return random.sample(self.templates, n)
        else:
            return []


RECIPE_TEMPLATES = [
    {
        "type": "clarify",
        "name": "食材澄清",
        "template": "你回答中提到的[食材]，这里是指[常见替代品]吗？如果不放[食材]，对成品有什么影响？",
        "slots": ["食材", "常见替代品"]
    },
    {
        "type": "clarify",
        "name": "术语澄清",
        "template": "你说到的[烹饪术语]具体是什么样的操作？对于初学者，有哪些简化替代手法？",
        "slots": ["烹饪术语"]
    },
    {
        "type": "reasoning",
        "name": "步骤探究",
        "template": "在[关键步骤]这一步中，如果跳过[操作]直接进行下一步，菜品的口感会发生什么变化？",
        "slots": ["关键步骤", "操作"]
    },
    {
        "type": "reasoning",
        "name": "技法对比",
        "template": "[技法A]和[技法B]都是常用的烹饪手法。这道菜为什么选用前者而非后者？两种技法各在什么场景下更合适？",
        "slots": ["技法A", "技法B"]
    },
    {
        "type": "reasoning",
        "name": "替代推理",
        "template": "[食材A]可以用什么食材替换？替换后这道菜的风味会发生怎样的变化？",
        "slots": ["食材A"]
    },
    {
        "type": "meta",
        "name": "安全评估",
        "template": "在制作这道菜的过程中，有哪些需要特别注意的安全事项（如油温控制、刀工安全）？初学者最常犯的错误是什么？",
        "slots": []
    },
    {
        "type": "meta",
        "name": "文化溯源",
        "template": "这道菜属于哪个菜系？为什么这个菜系的[口味特点]在这道菜中体现得特别明显？",
        "slots": ["口味特点"]
    },
    {
        "type": "meta",
        "name": "营养反思",
        "template": "从营养学角度看，这道菜的搭配是否均衡？如果要让它更健康，可以做什么调整而不破坏它的核心风味？",
        "slots": []
    },
]

POETRY_TEMPLATES = [
    {
        "type": "clarify",
        "name": "词义探析",
        "template": "你提到[关键词]，在这首诗的具体语境中，这个词应当如何理解？如果换成[近义词]，意蕴会有何变化？",
        "slots": ["关键词", "近义词"]
    },
    {
        "type": "clarify",
        "name": "背景澄清",
        "template": "你分析了这首诗的情感基调。诗人在创作此诗时处于什么样的处境和心境？这个背景信息是否支持你的解读？",
        "slots": []
    },
    {
        "type": "reasoning",
        "name": "意象推敲",
        "template": "[意象]在这句诗中象征着什么？从全诗的脉络来看，这个意象是如何与诗的整体情感基调呼应的？",
        "slots": ["意象"]
    },
    {
        "type": "reasoning",
        "name": "修辞分析",
        "template": "这句诗中诗人使用了[修辞手法]。请分析这种手法在此处的具体效果，并举例说明这种手法在其他诗作中的类似运用。",
        "slots": ["修辞手法"]
    },
    {
        "type": "reasoning",
        "name": "对比分析",
        "template": "这首诗和[同类诗作]在风格、主题或技法上有何异同？这种差异反映了诗人怎样的创作个性？",
        "slots": ["同类诗作"]
    },
    {
        "type": "meta",
        "name": "多重解读",
        "template": "对[诗句]的解读学界有不同的观点。你认同哪一种？请说明你的理由，并谈谈其他解读的合理之处和局限。",
        "slots": ["诗句"]
    },
    {
        "type": "meta",
        "name": "格律探究",
        "template": "从格律和音韵的角度分析这首诗的形式特点。诗人是否在某些地方突破了传统格律？如果突破了，用意何在？",
        "slots": []
    },
    {
        "type": "meta",
        "name": "影响溯源",
        "template": "这首诗受到了哪些前代诗人或作品的影响？它又对后世产生了怎样的影响？你认为它在中国诗歌史上的地位如何？",
        "slots": []
    },
]
