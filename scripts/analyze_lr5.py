"""
训练充分性诊断（MVE vs lr5配对+维度分解+Cohen's d）。来源：配对t+效应量

回答核心问题：训练充分后，菜谱 B vs A 是否从"不显著"变"显著"？
  → 显著=之前训练不足、方法有效；仍不显著=需换更大模型

数据要求：
  results/recipe_A.json,B,C          （MVE 版，已在）
  results/recipe_lr5_A.json,B,C      （lr5 版，下载后放入）

用法：python analyze_lr5.py

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import math
import statistics as st
from pathlib import Path

RES = Path("results")
DIMS = ["factual", "completeness", "logic", "depth"]


def load(name):
    p = RES / f"{name}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def paired(ya, yb):
    d = [b - a for a, b in zip(ya, yb)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n) if n else 0
    t = md / se if se > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    cohen = md / sd if sd > 0 else 0
    return md, t, p, cohen


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def group_stats(recs):
    ys = [r["Y"] for r in recs if r["Y"] is not None]
    ms = [r["M"] for r in recs if r.get("M")]
    dims = {}
    for dm in DIMS:
        vals = []
        for r in recs:
            pj = r["Y_detail"]["per_judge"]
            v = [jv[dm] for jv in pj.values() if jv and dm in jv]
            if v:
                vals.append(st.mean(v))
        dims[dm] = st.mean(vals) if vals else float("nan")
    return {"Y": st.mean(ys), "Ysd": st.pstdev(ys), "M": st.mean(ms) if ms else None,
            "dims": dims, "recs": recs}


def compare_versions(lines, tag, prefix):
    a, b, c = load(f"{prefix}_A"), load(f"{prefix}_B"), load(f"{prefix}_C")
    if not (a and b and c):
        lines.append(f"[{tag}] 数据不全，跳过\n")
        return None
    sa, sb, sc = group_stats(a), group_stats(b), group_stats(c)
    lines.append(f"======== {tag} ========")
    lines.append(f"  A: Y={sa['Y']:.3f}(sd={sa['Ysd']:.2f})   "
                 f"B: Y={sb['Y']:.3f}(sd={sb['Ysd']:.2f})   "
                 f"C: Y={sc['Y']:.3f}(sd={sc['Ysd']:.2f})")
    if sb["M"]:
        lines.append(f"  M:  B={sb['M']:.3f}  C={sc['M']:.3f}")
    # 配对检验（按question对齐）
    ia = {r["question"]: r["Y"] for r in a}
    ib = {r["question"]: r["Y"] for r in b}
    ic = {r["question"]: r["Y"] for r in c}
    common = sorted(set(ia) & set(ib) & set(ic))
    ya, yb, yc = [ia[q] for q in common], [ib[q] for q in common], [ic[q] for q in common]
    for nm, y1, y2 in [("B vs A", ya, yb), ("C vs A", ya, yc), ("C vs B", yb, yc)]:
        md, t, p, d = paired(y1, y2)
        lines.append(f"  {nm}: Δ={md:+.3f} p={p:.3f} {sig(p)} Cohen_d={d:+.3f}")
    lines.append("  维度(A/B/C): " + " | ".join(
        f"{dm}:{sa['dims'][dm]:.2f}/{sb['dims'][dm]:.2f}/{sc['dims'][dm]:.2f}" for dm in DIMS))
    lines.append("")
    return {"A": sa, "B": sb, "C": sc, "common": (ya, yb, yc)}


def main():
    lines = ["lr5 诊断分析：MVE vs lr5 对比", "=" * 44, ""]
    mve = compare_versions(lines, "MVE (lr=1e-6, 3轮)", "recipe")
    lr5 = compare_versions(lines, "lr5 (lr=5e-6+余弦, 5轮)", "recipe_lr5")

    if mve and lr5:
        lines.append("======== 关键判决：B vs A 是否随训练充分而改善 ========")
        for tag, res in [("MVE", mve), ("lr5", lr5)]:
            ya, yb, yc = res["common"]
            md, t, p, d = paired(ya, yb)
            lines.append(f"  {tag}: B-A Δ={md:+.3f} p={p:.3f} {sig(p)} d={d:+.3f}")
        lines.append("")
        lines.append("  判读：")
        lines.append("   - lr5的B-A显著(p<.05) & MVE不显著 → 训练不足是主因，方法有效 → 全量用lr5配置")
        lines.append("   - lr5的B-A仍不显著 → 不只训练问题，需换1.5B/3B(能力天花板)")
        lines.append("   - C组：看lr5后C是否仍显著劣于A/B（高强度反问过载是否真问题）")

    report = "\n".join(lines)
    Path("results/lr5_analysis_report.txt").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print("\n报告已保存至 results/lr5_analysis_report.txt")


if __name__ == "__main__":
    main()
