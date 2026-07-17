"""
T4重新分层（全局M三分位→low/mid/high三档输出）。来源：numpy百分位分割

从4候选打分结果中提取所有(问题,反问,M)对，
全局按M值排序，低/中/高各1/3，每档约500条（菜可跨档）。
呼应洞察：让M数据说话，不预设"哪种来源更好"。

产出：data/recipe/qa_pairs_clean/t4_low/mid/high.json（用于训练）
      results/t4_tertile_split.txt（分层统计）

用法：python t4_tertile_split.py
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import statistics as st
from pathlib import Path

GRADED = Path("results/t4_cq_graded.json")
OUT = Path("data/recipe/qa_pairs_clean")
BASE = Path("data/recipe/qa_pairs_clean/train.json")

graded = json.load(open(GRADED, encoding="utf-8"))
base = json.load(open(BASE, encoding="utf-8"))
base_idx = {r["question"]: r for r in base}

# 提取所有(问题,反问, M)三元组
triples = []
for g in graded:
    q = g["question"]
    cqs = g["cqs"]
    ms = g["M"]
    for tier in cqs:
        if ms.get(tier) is not None:
            triples.append((q, cqs[tier], ms[tier], tier))
triples.sort(key=lambda x: x[2])  # 按M升序
print(f"[T4-A] 共 {len(triples)} 个(问题,反问,M)三元组")

# 三分位
n = len(triples)
t1, t2 = n // 3, 2 * n // 3
tiers_data = {"low": triples[:t1], "mid": triples[t1:t2], "high": triples[t2:]}
lines = ["T4-A 全局M三分位分层", "=" * 30, f"n={n}, 三分位点: {triples[t1][2]:.3f}, {triples[t2][2]:.3f}", ""]

for label, ts in tiers_data.items():
    ms = [t[2] for t in ts]
    srcs = {}
    for t in ts:
        srcs[t[3]] = srcs.get(t[3], 0) + 1
    lines.append(f"{label}档: n={len(ts)} M均值={st.mean(ms):.3f} 范围[{min(ms):.2f},{max(ms):.2f}] 来源={srcs}")
    # 生成训练集
    recs = []
    for q, cq, m, src in ts:
        orig = base_idx.get(q)
        if not orig:
            continue
        r = dict(orig)
        r["counter_question"] = cq
        r["_tier"] = label
        r["_cq_M"] = m
        r["_cq_src"] = src
        recs.append(r)
    fp = OUT / f"t4_{label}.json"
    json.dump(recs, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines.append(f"  训练集: {len(recs)}条 → {fp}")

report = "\n".join(lines)
Path("results/t4_tertile_split.txt").write_text(report, encoding="utf-8")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
print(report)
print("\n[T4-A] 分层完成。训练命令(云端):")
print("for T in low mid high; do")
print("  python -m src.training.self_play --sft-model models/recipe_sft \\")
print("    --train-data data/recipe/qa_pairs_clean/t4_$T.json --domain recipe --group B \\")
print("    --iterations 5 --lr 5e-6 --lr-schedule cosine --train-limit 500 \\")
print("    --gen-batch 48 --ma-cache cache/t4_${T}_cache.json --api-key $KEY --out models/t4_$T")
print("done")
