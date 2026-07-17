"""
精简模式三方对比（原始/控长度/控长度+扣题）。来源：配对t检验，本实验T1条件
  精简：recipe_concise_{A,B,C}  (CONCISE_MODE, 150字)
  原始：recipe_lr5_{A,B,C}      (0.5B lr5, 无精简)

回答：控制长度后，反问组(B/C)的Y是否反超A？长度真的降了吗？M-Y关系如何？
若控长度后 B/C 追平或反超 A → 证明"A>B/C的表象主要是长度膨胀，反问本身不有害"。

用法：python analyze_t1.py

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import math
import statistics as st
from pathlib import Path

RES = Path("results")


def load(name):
    p = RES / f"{name}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def paired(y1, y2):
    d = [b - a for a, b in zip(y1, y2)]
    n = len(d)
    if n < 2:
        return 0, 1
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n) if n else 0
    t = md / se if se > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return md, p


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def stat_group(recs):
    Y = [r["Y"] for r in recs if r["Y"] is not None]
    L = [len(r["eval_answer"]) for r in recs]
    M = [r["M"] for r in recs if r.get("M")]
    return {"Y": st.mean(Y), "len": st.mean(L), "M": st.mean(M) if M else None,
            "qY": {r["question"]: r["Y"] for r in recs if r["Y"] is not None}}


def analyze(tag, prefix, lines):
    a, b, c = load(f"{prefix}_A"), load(f"{prefix}_B"), load(f"{prefix}_C")
    if not (a and b and c):
        lines.append(f"[{tag}] 数据不全({prefix})")
        return None
    sa, sb, sc = stat_group(a), stat_group(b), stat_group(c)
    lines.append(f"======== {tag} ({prefix}) ========")
    lines.append(f"  Y:   A={sa['Y']:.3f}  B={sb['Y']:.3f}  C={sc['Y']:.3f}")
    lines.append(f"  长度: A={sa['len']:.0f}  B={sb['len']:.0f}  C={sc['len']:.0f}")
    if sb["M"]:
        lines.append(f"  M:   B={sb['M']:.3f}  C={sc['M']:.3f}")
    # 配对检验
    common = sorted(set(sa["qY"]) & set(sb["qY"]) & set(sc["qY"]))
    ya = [sa["qY"][q] for q in common]
    yb = [sb["qY"][q] for q in common]
    yc = [sc["qY"][q] for q in common]
    for nm, y1, y2 in [("B-A", ya, yb), ("C-A", ya, yc), ("C-B", yb, yc)]:
        md, p = paired(y1, y2)
        lines.append(f"  {nm}: Δ={md:+.3f} p={p:.3f} {sig(p)}")
    lines.append("")
    return {"A": sa, "B": sb, "C": sc}


def main():
    lines = ["T1 分析：原始 vs 控长度 vs 控长度+扣题", "=" * 44, ""]
    orig = analyze("原始(0.5B lr5)", "recipe_lr5", lines)
    conc = analyze("控长度(CONCISE 150字)", "recipe_concise", lines)
    strict = analyze("控长度+扣题(STRICT)", "recipe_strict", lines)

    if orig and conc:
        lines.append("======== 控长度效应（控长度 - 原始）========")
        for g in "ABC":
            dY = conc[g]["Y"] - orig[g]["Y"]
            dL = conc[g]["len"] - orig[g]["len"]
            lines.append(f"  {g}: ΔY={dY:+.3f}  Δ长度={dL:+.0f}")
        lines.append("")

    if orig and strict:
        lines.append("======== 扣题防编造效应（控长度+扣题 - 原始）========")
        for g in "ABC":
            dY = strict[g]["Y"] - orig[g]["Y"]
            dL = strict[g]["len"] - orig[g]["len"]
            lines.append(f"  {g}: ΔY={dY:+.3f}  Δ长度={dL:+.0f}")
        lines.append("")

    def rank(s):
        ys = {g: s[g]["Y"] for g in "ABC"}
        return ">".join(sorted("ABC", key=lambda g: -ys[g]))

    lines.append("======== 三方格局对比 ========")
    if orig:
        lines.append(f"  原始:        {rank(orig)}  (A={orig['A']['Y']:.2f} B={orig['B']['Y']:.2f} C={orig['C']['Y']:.2f})")
    if conc:
        lines.append(f"  控长度:      {rank(conc)}  (A={conc['A']['Y']:.2f} B={conc['B']['Y']:.2f} C={conc['C']['Y']:.2f})")
    if strict:
        lines.append(f"  控长度+扣题: {rank(strict)}  (A={strict['A']['Y']:.2f} B={strict['B']['Y']:.2f} C={strict['C']['Y']:.2f})")
    lines.append("")
    lines.append("  判读：")
    lines.append("   - 控长度后 B/C 追平/反超 A → 长度是主因")
    lines.append("   - 控长度不够、但+扣题后 B/C 改善 → 编造/偏题是根本，需引导扣题(呼应'光控长度不够')")
    lines.append("   - 两者都不行 → 反问训练确有更深层损害，需查暴露偏差/损失设计")

    report = "\n".join(lines)
    Path("results/t1_analysis.txt").write_text(report, encoding="utf-8")
    try:
        print(report)
    except UnicodeEncodeError:
        import sys
        sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(report)
    print("\n报告已保存至 results/t1_analysis.txt")


if __name__ == "__main__":
    main()
