"""
机制假说检验（长度效应/Pearson相关/n-gram重复度）。来源：n-gram文本分析
  ⑥ 长度效应：各组答案长度分布 + 长度-Y相关（全组，非仅组内）
  ⑦ 格式污染/模式坍缩：答案重复模式、套话、n-gram重复度

猜想④（初答vs最终答）需训练时的初答，评测数据未保存，暂缺——列为待补。

用法：python probe_mechanism.py --prefixes recipe recipe_lr5 poetry

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

RES = Path("results")


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


def ngram_repeat_rate(text, n=4):
    """4-gram 重复率：1 - 去重后/总数。越高越重复。"""
    chars = re.sub(r"\s", "", text)
    if len(chars) < n + 1:
        return 0.0
    grams = [chars[i:i + n] for i in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def analyze(prefix, lines):
    groups = {g: load(f"{prefix}_{g}") for g in "ABC"}
    if not all(groups.values()):
        lines.append(f"[{prefix}] 数据不全，跳过\n")
        return
    lines.append(f"════════ {prefix} ════════")

    # ⑥ 长度
    lines.append("【⑥ 长度效应】")
    all_len, all_y = [], []
    for g in "ABC":
        L = [len(r["eval_answer"]) for r in groups[g]]
        Y = [r["Y"] for r in groups[g]]
        lines.append(f"  {g}: 长度均值={st.mean(L):.0f} 中位={st.median(L):.0f} sd={st.pstdev(L):.0f}")
        for l, y in zip(L, Y):
            if y is not None:
                all_len.append(l)
                all_y.append(y)
    r_all = pearson(all_len, all_y)
    lines.append(f"  全体 r(长度,Y)={r_all:+.3f}  "
                 + ("⚠️长度显著影响Y" if abs(r_all) > 0.25 else "长度影响有限"))
    # 分长度档看Y
    med = st.median(all_len)
    hi_y = st.mean([y for l, y in zip(all_len, all_y) if l >= med])
    lo_y = st.mean([y for l, y in zip(all_len, all_y) if l < med])
    lines.append(f"  长答案(≥{med:.0f}字) Y={hi_y:.3f}  短答案 Y={lo_y:.3f}  差={hi_y-lo_y:+.3f}")

    # ⑦ 重复模式
    lines.append("\n【⑦ 格式污染/重复度（4-gram重复率，越高越坏）】")
    for g in "ABC":
        rr = [ngram_repeat_rate(r["eval_answer"]) for r in groups[g]]
        # 高重复答案数
        high = sum(1 for x in rr if x > 0.15)
        lines.append(f"  {g}: 平均重复率={st.mean(rr):.3f}  高重复(>0.15)答案数={high}/{len(rr)}")
    # 重复率 vs Y 相关
    rr_all, y2 = [], []
    for g in "ABC":
        for r in groups[g]:
            if r["Y"] is not None:
                rr_all.append(ngram_repeat_rate(r["eval_answer"]))
                y2.append(r["Y"])
    rc = pearson(rr_all, y2)
    lines.append(f"  全体 r(重复率,Y)={rc:+.3f}  "
                 + ("⚠️重复拖累Y" if rc < -0.2 else "重复影响有限"))
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", nargs="+", default=["recipe", "recipe_lr5", "poetry"])
    args = ap.parse_args()
    lines = ["机制假说检验 ⑥长度 / ⑦格式污染", "=" * 44,
             "注：④初答vs最终答 需训练时初答数据，评测未存，待补", ""]
    for p in args.prefixes:
        analyze(p, lines)
    report = "\n".join(lines)
    Path("results/probe_mechanism.txt").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print("\n报告已保存至 results/probe_mechanism.txt")


if __name__ == "__main__":
    main()
