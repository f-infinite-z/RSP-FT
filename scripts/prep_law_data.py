"""
法律数据准备（train/val/test划分+MVE子集+剔除超短答案）。来源：标准数据划分方法
输入：data/law/qa_pairs/all.json (2964条)
输出：data/law/qa_pairs_clean/{train,val,test,train_mve,test_mve}.json
可复现 seed=42。剔除超短答案(<20字)。
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, random
from pathlib import Path

random.seed(42)
d = json.load(open("data/law/qa_pairs/all.json", encoding="utf-8"))

# 剔除超短答案(<20字，多为无效)
clean = [r for r in d if len(r.get("answer", "")) >= 20 and r.get("counter_question")]
print("原始 %d → 剔除超短后 %d" % (len(d), len(clean)))

random.shuffle(clean)
n = len(clean)
# 划分：与菜谱比例对齐(train大头/val少量/test)
n_test = 300
n_val = 200
test = clean[:n_test]
val = clean[n_test:n_test+n_val]
train = clean[n_test+n_val:]

out = Path("data/law/qa_pairs_clean")
out.mkdir(parents=True, exist_ok=True)
json.dump(train, open(out/"train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(val, open(out/"val.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(test, open(out/"test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# MVE 子集（对齐菜谱：train前500 / test前100）
train_mve = train[:500]
test_mve = test[:100]
json.dump(train_mve, open(out/"train_mve.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(test_mve, open(out/"test_mve.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("train=%d val=%d test=%d" % (len(train), len(val), len(test)))
print("train_mve=%d test_mve=%d" % (len(train_mve), len(test_mve)))
print("输出目录:", out)
# 编分布检查
from collections import Counter
print("train_mve 编分布:", dict(Counter(r.get("law_bian","?") for r in train_mve)))
print("test_mve 编分布:", dict(Counter(r.get("law_bian","?") for r in test_mve)))
