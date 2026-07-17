"""
MVE组间配对t检验+Cohen's d+裁判区分度。来源：Student(1908)+Cohen(1988)标准公式

对每个领域做：
  - 三组 Y 的均值/标准差
  - B vs A / C vs A / C vs B 的配对检验（配对 t，正态近似 p）+ Cohen's d
  - 三裁判打分区分度（均值/sd/范围）
  - 整体 Y 分布直方

用法：python stat_test.py [--domains poetry recipe]
仅依赖标准库 + json。

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

RES = Path("results")


def load(name):
    return json.load(open(RES / f"{name}.json", encoding="utf-8"))


def paired_test(ya, yb):
    diffs = [b - a for a, b in zip(ya, yb)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0
    se = sd_d / math.sqrt(n) if n else 0
    t = mean_d / se if se > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    d = mean_d / sd_d if sd_d > 0 else 0
    return mean_d, t, p, d


def sig_mark(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def analyze_domain(dom, out_lines):
    a, b, c = load(f"{dom}_A"), load(f"{dom}_B"), load(f"{dom}_C")
    ia = {r["question"]: r["Y"] for r in a}
    ib = {r["question"]: r["Y"] for r in b}
    ic = {r["question"]: r["Y"] for r in c}
    common = sorted(set(ia) & set(ib) & set(ic))
    ya = [ia[q] for q in common]
    yb = [ib[q] for q in common]
    yc = [ic[q] for q in common]

    out_lines.append(f"======== {dom} (n={len(common)}) ========")
    out_lines.append(f"  A: mean={st.mean(ya):.3f} sd={st.pstdev(ya):.3f}")
    out_lines.append(f"  B: mean={st.mean(yb):.3f} sd={st.pstdev(yb):.3f}")
    out_lines.append(f"  C: mean={st.mean(yc):.3f} sd={st.pstdev(yc):.3f}")
    for name, y1, y2 in [("B vs A", ya, yb), ("C vs A", ya, yc), ("C vs B", yb, yc)]:
        md, t, p, d = paired_test(y1, y2)
        out_lines.append(f"  {name}: Δ={md:+.3f}  t={t:+.2f}  p={p:.3f} {sig_mark(p)}  Cohen_d={d:+.3f}")

    # 裁判区分度
    out_lines.append("  --- 裁判区分度 ---")
    allrecs = a + b + c
    for judge in ["deepseek", "doubao", "qwen"]:
        ov = [r["Y_detail"]["per_judge"][judge]["overall"]
              for r in allrecs if r["Y_detail"]["per_judge"].get(judge)]
        if ov:
            out_lines.append(f"    {judge}: 均值={st.mean(ov):.2f} sd={st.pstdev(ov):.2f} "
                             f"范围[{min(ov):.1f},{max(ov):.1f}]")
    ys = [r["Y"] for r in allrecs if r["Y"] is not None]
    buckets = collections.Counter(round(y) for y in ys)
    out_lines.append("    整体Y分布(取整): " + " ".join(f"{k}:{v}" for k, v in sorted(buckets.items())))
    out_lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["poetry", "recipe"])
    ap.add_argument("--out", default="results/stat_test_report.txt")
    args = ap.parse_args()

    lines = ["MVE 组间差异统计检验报告", "=" * 40, ""]
    for dom in args.domains:
        if (RES / f"{dom}_A.json").exists():
            analyze_domain(dom, lines)
        else:
            lines.append(f"[跳过] {dom}: 无数据\n")

    report = "\n".join(lines)
    Path(args.out).write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print(f"\n报告已保存至 {args.out}")


if __name__ == "__main__":
    main()
