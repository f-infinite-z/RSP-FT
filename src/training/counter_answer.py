"""
M_A 反问回答策略

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

# ============================================================
# 人工批注（作者）
# ============================================================
# 工程基础设施，让M_A仿真补充的时候有充足的信息量可以跑起来，作用是模拟用户提供上下文，让小模型能补充足信息，M_A全程训练是冻结的，而且最后测试的是M_B。防止小模型补充的全是占位空话。
# 四级回退机制：预生成缓存（sha1精确查表，零API等待）→检索匹配（训练集中找语义最接近的反问）→LLM生成（实时调API）→占位兜底。目的是为了提供上下文，让对话更真实，但不会告诉B怎么答。
# 这不算引入外部信息，SPIN (Chen et al., ICML 2024) 已证明用外部信号引导自博弈是标准做法，前提是训练主体（M_B）自己产出最终答案。
# 预生成缓存优先级最高，主要还是为了优化速度，直接查表很快，可以提前并行预生成。可以用便宜的模型先生成好，反正是引导作用上面讲了不算作弊，LLM实时调的话有的等。如果认为预生成可能存在隐藏变量的话可以尝试去掉这一环节。
# ============================================================

M_A 在实验组轮2中作为“提问者”需回答 M_B 的反问，采用三级回退：
  1. 检索匹配：反问嵌入后在训练集 QA 的 counter_question 中检索相似度>阈值，取对应 counter_answer
  2. 大模型生成：无匹配时调用 DeepSeek/GPT 依据反问生成回答
  3. 占位填充：API 不可用时使用通用模板

M_A 本身不参与梯度更新，故引入外部知识不构成作弊；
需在论文中报告各级触发次数（本模块内置计数）。
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SIM_THRESHOLD = 0.8

GENERIC_TEMPLATE = "这是一个很好的问题。关于「{kw}」，需要结合具体情境来理解，其核心在于把握关键要素之间的关系。"


@dataclass
class RetrievalItem:
    counter_question: str
    counter_answer: str


class CounterAnswerer:
    """三级回退的反问回答器，带调用统计。"""

    def __init__(self,
                 retrieval_pool: Optional[List[RetrievalItem]] = None,
                 embed_fn=None,
                 llm_client=None,
                 llm_model: str = "deepseek-v4-flash",
                 domain: str = "",
                 sim_threshold: float = SIM_THRESHOLD,
                 cache_path: Optional[str] = None):
        self.pool = retrieval_pool or []
        self.embed_fn = embed_fn
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.domain = domain
        self.sim_threshold = sim_threshold
        self._pool_embs = None
        self.stats = {"cache": 0, "retrieval": 0, "llm": 0, "placeholder": 0}

        # 预生成补充库（全量加速）：key=sha1(question|||cq) -> 补充r
        self._cache = {}
        if cache_path:
            import json as _json
            import os
            if os.path.exists(cache_path):
                try:
                    self._cache = _json.load(open(cache_path, encoding="utf-8"))
                    print(f"[CounterAnswerer] 载入预生成缓存 {len(self._cache)} 条: {cache_path}",
                          flush=True)
                except Exception as e:
                    print(f"[CounterAnswerer] 缓存载入失败({e})，回退实时生成", flush=True)

        if self.embed_fn and self.pool:
            self._build_index()

    def _try_cache(self, question: str, counter_question: str) -> Optional[str]:
        if not self._cache:
            return None
        import hashlib
        key = hashlib.sha1((question + "|||" + counter_question).encode("utf-8")).hexdigest()
        return self._cache.get(key)

    def _build_index(self):
        import numpy as np
        embs = self.embed_fn([it.counter_question for it in self.pool])
        embs = np.asarray(embs, dtype="float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._pool_embs = embs / norms

    # ---- 三级策略 ----

    def _try_retrieval(self, counter_question: str) -> Optional[str]:
        if self._pool_embs is None or not self.embed_fn:
            return None
        import numpy as np
        q = np.asarray(self.embed_fn([counter_question])[0], dtype="float32")
        n = np.linalg.norm(q)
        if n == 0:
            return None
        q = q / n
        sims = self._pool_embs @ q
        best = int(sims.argmax())
        if sims[best] >= self.sim_threshold:
            return self.pool[best].counter_answer
        return None

    def _try_llm(self, question: str, counter_question: str,
                 timeout: float = 30.0, retries: int = 2) -> Optional[str]:
        if self.llm_client is None:
            return None
        dom = {"recipe": "中华菜谱与烹饪", "poetry": "古诗词鉴赏", "law": "中国民法典"}.get(self.domain, self.domain)
        sys_p = f"你是{dom}领域专家。请简洁、准确地回答下面的追问，2-4句话。"
        usr = f"背景问题：{question}\n追问：{counter_question}"
        import time
        for attempt in range(retries + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.llm_model, temperature=0.5, max_tokens=400,
                    timeout=timeout,
                    messages=[{"role": "system", "content": sys_p},
                              {"role": "user", "content": usr}])
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                print(f"    [M_A] LLM调用失败({type(e).__name__})，回退占位", flush=True)
                return None

    def _placeholder(self, counter_question: str) -> str:
        kw = counter_question.strip().rstrip("？?").split("，")[-1][:12] or "该问题"
        return GENERIC_TEMPLATE.format(kw=kw)

    def answer(self, question: str, counter_question: str) -> str:
        # 0级：预生成缓存（全量加速，命中则零 API 等待）
        ans = self._try_cache(question, counter_question)
        if ans is not None:
            self.stats["cache"] += 1
            return ans
        ans = self._try_retrieval(counter_question)
        if ans is not None:
            self.stats["retrieval"] += 1
            return ans
        ans = self._try_llm(question, counter_question)
        if ans is not None:
            self.stats["llm"] += 1
            return ans
        self.stats["placeholder"] += 1
        return self._placeholder(counter_question)

    def report(self) -> Dict[str, int]:
        total = sum(self.stats.values())
        out = dict(self.stats)
        out["total"] = total
        if total:
            out["retrieval_rate"] = round(self.stats["retrieval"] / total, 3)
            out["llm_rate"] = round(self.stats["llm"] / total, 3)
            out["placeholder_rate"] = round(self.stats["placeholder"] / total, 3)
        return out


def build_pool_from_dataset(path: str) -> List[RetrievalItem]:
    """从训练集 json 构造检索池（使用 counter_question/counter_answer 字段）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pool = []
    for x in data:
        cq = str(x.get("counter_question", "")).strip()
        ca = str(x.get("counter_answer", "")).strip()
        if cq and ca:
            pool.append(RetrievalItem(counter_question=cq, counter_answer=ca))
    return pool
