"""
隐藏变量探查（长度混淆/维度崩溃/退化率/裁判分歧/M-Y连续OLS）。来源：OLS回归标准方法

候选隐藏变量：
  1. 答案长度：反问组答案是否更长/更短？长度与Y是否相关（长度混淆）？
  2. 维度崩溃定位：B/C组暴跌是哪个维度导致（factual/completeness/logic/depth）？
  3. 退化率：是否有答案截断/复读/空答（生成质量问题）？
  4. 裁判分歧：某裁判是否异常拉低某组？
  5. 版本对比：MVE vs lr5，训练充分后各维度怎么变（恶化还是改善）？

用法：python probe_confounders.py --prefix recipe_lr5
      python probe_confounders.py --prefix recipe        (MVE版)

来源：论文实验代码。
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
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


def analyze(prefix, lines):
    groups = {g: load(f"{prefix}_{g}") for g in "ABC"}
    if not all(groups.values()):
        lines.append(f"[{prefix}] 数据不全，跳过\n")
        return
    lines.append(f"════════ 隐藏变量探查：{prefix} ════════\n")

    # 1. 答案长度
    lines.append("【1. 答案长度】")
    lengths = {}
    for g in "ABC":
        L = [len(r["eval_answer"]) for r in groups[g]]
        lengths[g] = L
        lines.append(f"  {g}: 均值={st.mean(L):.0f}字 中位={st.median(L):.0f} "
                     f"范围[{min(L)},{max(L)}] sd={st.pstdev(L):.0f}")

    # 2. 长度 vs Y 相关（长度混淆检验）
    lines.append("\n【2. 长度-Y相关（每组内，检验长度混淆）】")
    for g in "ABC":
        L = [len(r["eval_answer"]) for r in groups[g]]
        Y = [r["Y"] for r in groups[g]]
        pairs = [(l, y) for l, y in zip(L, Y) if y is not None]
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        note = "⚠️长度可能混淆" if abs(r) > 0.3 else "长度影响小"
        lines.append(f"  {g}: r(长度,Y)={r:+.3f}  {note}")

    # 3. 维度崩溃定位（相对A的变化）
    lines.append("\n【3. 维度分解（相对A组的变化，定位崩在哪）】")
    def dim_mean(recs, dm):
        vals = []
        for r in recs:
            pj = r["Y_detail"]["per_judge"]
            v = [jv[dm] for jv in pj.values() if jv and dm in jv]
            if v:
                vals.append(st.mean(v))
        return st.mean(vals) if vals else float("nan")
    base = {dm: dim_mean(groups["A"], dm) for dm in DIMS}
    lines.append(f"  A基线: " + "  ".join(f"{dm}={base[dm]:.2f}" for dm in DIMS))
    for g in "BC":
        deltas = {dm: dim_mean(groups[g], dm) - base[dm] for dm in DIMS}
        worst = min(deltas, key=deltas.get)
        lines.append(f"  {g}-A: " + "  ".join(f"{dm}={deltas[dm]:+.2f}" for dm in DIMS)
                     + f"   [最崩:{worst}]")

    # 4. 退化率（答案质量问题）
    lines.append("\n【4. 退化答案率（<15字/低多样性，生成质量）】")
    for g in "ABC":
        degen = sum(1 for r in groups[g]
                    if len(r["eval_answer"]) < 15 or len(set(r["eval_answer"])) < 10)
        lines.append(f"  {g}: {degen}/{len(groups[g])}")

    # 5. 裁判分歧（各裁判对各组的均分）
    lines.append("\n【5. 裁判×组别（看某裁判是否异常拉低某组）】")
    for judge in ["deepseek", "doubao", "qwen"]:
        row = []
        for g in "ABC":
            ov = [r["Y_detail"]["per_judge"][judge]["overall"]
                  for r in groups[g] if r["Y_detail"]["per_judge"].get(judge)]
            row.append(st.mean(ov) if ov else float("nan"))
        lines.append(f"  {judge:<10}: A={row[0]:.2f} B={row[1]:.2f} C={row[2]:.2f}")

    # 6. 连续 M-Y 回归（抛开A/B/C标签，看反问质量真实效应）— P8指向
    lines.append("\n【6. 连续 M-Y 回归（抛开分组标签，仅B/C有反问的条目）】")
    my = []
    for g in "BC":
        for r in groups[g]:
            m, y = r.get("M"), r.get("Y")
            if m and y is not None:
                my.append((float(m), float(y)))
    if len(my) >= 5:
        ms = [p[0] for p in my]
        ys = [p[1] for p in my]
        r = pearson(ms, ys)
        # OLS 斜率
        n = len(my)
        mm, my_ = sum(ms) / n, sum(ys) / n
        slope = sum((x - mm) * (y - my_) for x, y in zip(ms, ys)) / sum((x - mm) ** 2 for x in ms)
        lines.append(f"  n={n} (B+C所有有反问条目)")
        lines.append(f"  r(M,Y)={r:+.3f}  斜率={slope:+.3f}")
        note = "反问质量↑→回答质量↑（正向真实效应）" if r > 0.15 else \
               ("反问质量↑→回答质量↓（负向）" if r < -0.15 else "M-Y 关联弱")
        lines.append(f"  → {note}")
        # 分M高低两半看Y
        med = st.median(ms)
        hi = [y for m, y in my if m >= med]
        lo = [y for m, y in my if m < med]
        lines.append(f"  高M组(M≥{med:.1f}) Y均值={st.mean(hi):.3f}  低M组 Y均值={st.mean(lo):.3f}  "
                     f"差={st.mean(hi)-st.mean(lo):+.3f}")
    else:
        lines.append("  数据不足")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="recipe_lr5")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    lines = []
    analyze(args.prefix, lines)
    report = "\n".join(lines)
    out = args.out or f"results/probe_{args.prefix}.txt"
    Path(out).write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print(f"\n报告已保存至 {out}")


if __name__ == "__main__":
    main()
