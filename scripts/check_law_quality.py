"""
法律QA质量抽查（题型分布+法条覆盖+编分布+长度统计）。来源：标准质检方法
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, random, statistics as st
from collections import Counter
from pathlib import Path

d = json.load(open("data/law/qa_pairs/all.json", encoding="utf-8"))
L = []
def log(s=""): L.append(str(s))

log("=" * 60)
log("法律QA（民法典）质量抽查")
log("=" * 60)
log("总条数: %d" % len(d))
log("字段: %s" % list(d[0].keys()))

# 题型分布
log("\n【题型分布】")
for t, n in Counter(r.get("qa_type","?") for r in d).most_common():
    log("  %s: %d" % (t, n))

# 覆盖法条数
tiaos = set(r.get("law_tiao","") for r in d)
log("\n覆盖法条数: %d" % len(tiaos))

# 编分布
log("\n【编分布】")
for b, n in Counter(r.get("law_bian","?") for r in d).most_common():
    log("  %s: %d" % (b, n))

# 长度统计
qlen = [len(r.get("question","")) for r in d]
alen = [len(r.get("answer","")) for r in d]
log("\n问题长度: 均值%d 范围[%d,%d]" % (st.mean(qlen), min(qlen), max(qlen)))
log("答案长度: 均值%d 范围[%d,%d]" % (st.mean(alen), min(alen), max(alen)))

# 可疑样本检测
log("\n【可疑样本检测】")
short_a = [r for r in d if len(r.get("answer",""))<20]
short_q = [r for r in d if len(r.get("question",""))<8]
no_cq = [r for r in d if not r.get("counter_question")]
# 答案是否引用了本条(粗查:答案含"本条"或与law_source有重叠)
log("  超短答案(<20字): %d" % len(short_a))
log("  超短问题(<8字): %d" % len(short_q))
log("  缺反问: %d" % len(no_cq))

# 随机抽5条人工看
random.seed(7)
log("\n【随机抽查5条】")
for r in random.sample(d, 5):
    log("-" * 50)
    log("[%s|%s] 题型:%s" % (r.get("law_bian",""), r.get("law_tiao",""), r.get("qa_type","")))
    log("问: %s" % r.get("question","")[:80])
    log("答: %s" % r.get("answer","")[:150])
    log("反问: %s" % str(r.get("counter_question",""))[:60])
    log("原文(核验用): %s" % r.get("law_source","")[:80])

Path("_law_check.txt").write_text("\n".join(L), encoding="utf-8")
print("done")
