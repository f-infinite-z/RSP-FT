"""
性价比分析（审稿意见6）：反问边际收益 / 推理成本
作者：臧恒杰（2026-08-29）
成本口径：生成答案的输出 token 数（中文按 1 字≈1 token 粗估）+ 反问环节额外 token
输出：results/cost_benefit_report.txt
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

GROUPS = {
    "0.5B":   ["recipe_lr5_{G}.json"],
    "1.5B":   ["recipe_1.5b_{G}.json"],
    "3B":     ["recipe_3b_{G}.json"],
    "poetry": ["poetry_{G}.json"],
    "law":    ["law_{G}.json"],
}


def load(name, g):
    p = ROOT / "results" / name.format(G=g)
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def main():
    lines = []
    add = lines.append
    add("=" * 76)
    add("性价比分析（意见6）：反问边际收益 / 推理成本")
    add("=" * 76)
    add("\n【一】输出成本（生成答案平均字符数，1字≈1 token 粗估）")
    add(f"{'规模/领域':<10} {'A组':>7} {'B组':>7} {'C组':>7} {'B额外':>7} {'C额外':>8}")
    lens = {}
    ys = {}
    for tag, pats in GROUPS.items():
        l, y = {}, {}
        for g in ["A", "B", "C"]:
            rows = load(pats[0], g)
            if rows is None:
                continue
            l[g] = sum(len(r.get("eval_answer") or "") for r in rows) / len(rows)
            y[g] = sum(r["Y"] or 0 for r in rows) / len(rows)
        lens[tag], ys[tag] = l, y
        if "A" in l:
            extra_b = (l["B"] - l["A"]) / l["A"] if l["A"] else 0
            extra_c = (l["C"] - l["A"]) / l["A"] if l.get("C") and l["A"] else float("nan")
            add(f"{tag:<10} {l.get('A',0):>7.0f} {l.get('B',0):>7.0f} {l.get('C',0):>7.0f} "
                f"{extra_b*100:>6.0f}% {extra_c*100:>7.0f}%")

    add("\n【二】反问边际收益/推理成本（ΔY ÷ 额外token比例）")
    add(f"{'规模/领域':<10} {'ΔY(B-A)':>9} {'ΔY(C-A)':>9} {'ΔY(C-B)':>9} "
        f"{'收益/成本(B-A)':>14} {'收益/成本(C-B)':>14}")
    for tag in GROUPS:
        l, y = lens[tag], ys[tag]
        if "A" not in l or "B" not in l:
            continue
        dy_ba = y["B"] - y["A"]
        extra_b = (l["B"] - l["A"]) / l["A"] if l["A"] else 0
        rc_ba = dy_ba / extra_b if extra_b else float("nan")
        if "C" in l:
            dy_ca = y["C"] - y["A"]
            dy_cb = y["C"] - y["B"]
            extra_cb = (l["C"] - l["B"]) / l["B"] if l["B"] else 0
            rc_cb = dy_cb / extra_cb if extra_cb else float("nan")
        else:
            dy_ca = dy_cb = rc_cb = float("nan")
        add(f"{tag:<10} {dy_ba:>9.3f} {dy_ca:>9.3f} {dy_cb:>9.3f} "
            f"{rc_ba:>14.3f} {rc_cb:>14.3f}")

    add("\n【三】判读")
    add(" - 收益/成本为负 → 反问带来额外推理成本且质量下降（当前小模型反问设计的现状）")
    add(" - 绝对值越小 → 反问性价比越差；CQLS/控长度等改进若能收窄 ΔY 即提升性价比")
    add(" - 延迟敏感场景指导：仅当 ΔY 显著为正且额外延迟可接受时才建议启用反问")
    txt = "\n".join(lines)
    out = ROOT / "results" / "cost_benefit_report.txt"
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
