"""
裁判视角调节（裁判×组别交叉表+维度分歧量化）。来源：交互效应分析标准方法

回答：不同专业取向的裁判（deepseek逻辑/doubao文学/qwen知识），
      对 A/B/C 三组的评价是否系统性不同？"反问是否有益"的结论是否依赖裁判视角？

不 cherry-pick，而是显式报告各裁判视角下的结论差异，把分歧当发现。

用法：python judge_perspective.py

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import statistics as st
from pathlib import Path

RES = Path("results")
JUDGES = ["deepseek", "doubao", "qwen"]
JUDGE_CN = {"deepseek": "DeepSeek(逻辑/事实)", "doubao": "豆包(文学/赏析)", "qwen": "Qwen(知识)"}
DIMS = ["factual", "completeness", "logic", "depth"]


def load(f):
    return json.load(open(RES / f"{f}.json", encoding="utf-8"))


def judge_overall(rec, judge):
    """某条记录中，某裁判的 overall 分（无则 None）。"""
    pj = rec.get("Y_detail", {}).get("per_judge", {})
    sc = pj.get(judge)
    return sc.get("overall") if sc else None


def analyze_domain(dom, lines):
    groups = {g: load(f"{dom}_{g}") for g in "ABC"}
    lines.append(f"======== {dom} · 裁判×组别 交叉表（overall 均值）========")
    header = f"{'裁判':<22}" + "".join(f"{'  '+g+'组':>10}" for g in "ABC") + f"{'  B-A':>9}{'  C-A':>9}"
    lines.append(header)

    for judge in JUDGES:
        row_means = {}
        for g in "ABC":
            vals = [judge_overall(r, judge) for r in groups[g]]
            vals = [v for v in vals if v is not None]
            row_means[g] = st.mean(vals) if vals else float("nan")
        ba = row_means["B"] - row_means["A"]
        ca = row_means["C"] - row_means["A"]
        lines.append(f"{JUDGE_CN[judge]:<22}" +
                     "".join(f"{row_means[g]:>10.2f}" for g in "ABC") +
                     f"{ba:>+9.2f}{ca:>+9.2f}")

    # 三裁判对"反问是否有益"(B-A)的方向是否一致
    lines.append("")
    lines.append("  【视角分歧检验】各裁判 B-A（反问适度增益）方向：")
    directions = []
    for judge in JUDGES:
        vals_a = [judge_overall(r, judge) for r in groups["A"]]
        vals_b = [judge_overall(r, judge) for r in groups["B"]]
        ma = st.mean([v for v in vals_a if v is not None])
        mb = st.mean([v for v in vals_b if v is not None])
        d = mb - ma
        directions.append(d > 0)
        arrow = "正向↑" if d > 0 else "负向↓"
        lines.append(f"    {JUDGE_CN[judge]}: B-A={d:+.3f} ({arrow})")
    if all(directions):
        lines.append("    → 三裁判一致认为适度反问(B)正向")
    elif not any(directions):
        lines.append("    → 三裁判一致认为适度反问(B)负向")
    else:
        lines.append("    → ⚠️ 裁判间方向分歧！结论依赖裁判视角（支持'裁判视角调节'假说）")

    # 维度层面：各裁判在 factual vs depth 上对反问的态度
    lines.append("")
    lines.append("  【维度视角】各裁判对 C 组(高强度反问)的 factual 与 depth 变化(相对A)：")
    for judge in JUDGES:
        def dim_mean(g, dm):
            vs = []
            for r in groups[g]:
                sc = r["Y_detail"]["per_judge"].get(judge)
                if sc and dm in sc:
                    vs.append(sc[dm])
            return st.mean(vs) if vs else float("nan")
        fac_d = dim_mean("C", "factual") - dim_mean("A", "factual")
        dep_d = dim_mean("C", "depth") - dim_mean("A", "depth")
        lines.append(f"    {JUDGE_CN[judge]}: Δfactual={fac_d:+.2f}  Δdepth={dep_d:+.2f}")
    lines.append("")


def main():
    lines = ["裁判视角调节分析（方案B）", "=" * 44, ""]
    for dom in ["poetry", "recipe"]:
        if (RES / f"{dom}_A.json").exists():
            analyze_domain(dom, lines)
        else:
            lines.append(f"[跳过] {dom}: 无数据\n")
    report = "\n".join(lines)
    Path("results/judge_perspective_report.txt").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print("\n报告已保存至 results/judge_perspective_report.txt")


if __name__ == "__main__":
    main()
