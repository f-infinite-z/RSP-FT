"""
容量假说判读（0.5B vs 1.5B M-Y相关+配对检验+方差对比）。来源：Pearson+配对t

回答：更大模型是否缓解"主任务挤占"？
  - M-Y相关增强 + A/B/C差异改善 → 容量缓解挤占（乐观：方法在大模型有前途）
  - M-Y相关仍弱 + A/B/C仍差 → 挤占非容量问题（本质在长度/损失/暴露偏差）

数据要求：
  results/recipe_lr5_{A,B,C}.json      （0.5B, lr5e-6+余弦+5轮）
  results/recipe_1.5b_{A,B,C}.json     （1.5B, 同配置；T7产出后放入）

用法：python analyze_capacity.py

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


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def paired(y1, y2):
    d = [b - a for a, b in zip(y1, y2)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n) if n else 0
    t = md / se if se > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return md, p


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def analyze(prefix, lines):
    a, b, c = load(f"{prefix}_A"), load(f"{prefix}_B"), load(f"{prefix}_C")
    if not (a and b and c):
        return None
    # 各组Y/M
    stat = {}
    for g, d in [("A", a), ("B", b), ("C", c)]:
        Y = [r["Y"] for r in d if r["Y"] is not None]
        M = [r["M"] for r in d if r.get("M")]
        L = [len(r["eval_answer"]) for r in d]
        stat[g] = {"Y": st.mean(Y), "M": st.mean(M) if M else None,
                   "len": st.mean(L), "qY": {r["question"]: r["Y"] for r in d if r["Y"] is not None}}
    # 连续 M-Y 相关（B+C）
    my = []
    for d in (b, c):
        for r in d:
            if r.get("M") and r["Y"] is not None:
                my.append((r["M"], r["Y"]))
    r_my = pearson([p[0] for p in my], [p[1] for p in my])
    # 配对检验
    common = sorted(set(stat["A"]["qY"]) & set(stat["B"]["qY"]) & set(stat["C"]["qY"]))
    ya = [stat["A"]["qY"][q] for q in common]
    yb = [stat["B"]["qY"][q] for q in common]
    yc = [stat["C"]["qY"][q] for q in common]
    _, p_ba = paired(ya, yb)
    _, p_ca = paired(ya, yc)
    _, p_cb = paired(yb, yc)
    return {"stat": stat, "r_my": r_my, "n_my": len(my),
            "p_ba": p_ba, "p_ca": p_ca, "p_cb": p_cb,
            "ya": ya, "yb": yb, "yc": yc}


def main():
    lines = ["容量假说判读：0.5B vs 1.5B (相同配置 lr5e-6+余弦+5轮)", "=" * 56, ""]
    r05 = analyze("recipe_lr5", lines)
    r15 = analyze("recipe_1.5b", lines)

    if not r05:
        lines.append("[缺] 0.5B (recipe_lr5) 数据")
    if not r15:
        lines.append("[缺] 1.5B (recipe_1.5b) 数据——T7完成后重跑本脚本")

    for tag, res in [("0.5B", r05), ("1.5B", r15)]:
        if not res:
            continue
        s = res["stat"]
        lines.append(f"======== {tag} ========")
        lines.append(f"  Y:  A={s['A']['Y']:.3f}  B={s['B']['Y']:.3f}  C={s['C']['Y']:.3f}")
        lines.append(f"  M:  B={s['B']['M']:.3f}  C={s['C']['M']:.3f}" if s['B']['M'] else "  M: —")
        lines.append(f"  长度: A={s['A']['len']:.0f}  B={s['B']['len']:.0f}  C={s['C']['len']:.0f}")
        lines.append(f"  连续 r(M,Y)={res['r_my']:+.3f} (n={res['n_my']})")
        lines.append(f"  配对: B-A p={res['p_ba']:.3f}{sig(res['p_ba'])}  "
                     f"C-A p={res['p_ca']:.3f}{sig(res['p_ca'])}  "
                     f"C-B p={res['p_cb']:.3f}{sig(res['p_cb'])}")
        lines.append("")

    # 核心判读
    if r05 and r15:
        lines.append("======== 容量假说判读 ========")
        dr = r15["r_my"] - r05["r_my"]
        lines.append(f"  M-Y相关变化: 0.5B={r05['r_my']:+.3f} → 1.5B={r15['r_my']:+.3f}  (Δ={dr:+.3f})")
        if dr > 0.1:
            lines.append("  → M-Y相关【增强】：支持容量缓解挤占（大模型让反问质量效应更显著）")
        elif dr < -0.1:
            lines.append("  → M-Y相关【减弱】：容量未缓解，反而更弱")
        else:
            lines.append("  → M-Y相关基本不变：容量非主导因素（挤占本质在长度/损失/暴露偏差）")
        # A/B/C格局
        def pattern(res):
            s = res["stat"]
            ys = {g: s[g]["Y"] for g in "ABC"}
            return " > ".join(sorted("ABC", key=lambda g: -ys[g]))
        lines.append(f"  A/B/C格局: 0.5B [{pattern(r05)}]  →  1.5B [{pattern(r15)}]")
        lines.append(f"  C组相对A: 0.5B C-A p={r05['p_ca']:.3f} → 1.5B C-A p={r15['p_ca']:.3f}")
        lines.append("  判读指引：")
        lines.append("   - 1.5B的C组不再显著低于A(p>0.05) + M-Y相关增强 → 容量缓解挤占✅")
        lines.append("   - 1.5B仍C<A显著 + M-Y仍弱 → 挤占非容量问题，需控长度/改损失")

    report = "\n".join(lines)
    Path("results/capacity_analysis.txt").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print("\n报告已保存至 results/capacity_analysis.txt")


if __name__ == "__main__":
    main()
