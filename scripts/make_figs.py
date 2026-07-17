"""
MVE三组Y柱状+四维分解+箱线+跨领域对照图。来源：matplotlib
原创RSP-FT训练代码为作者独立实现（见 src/training/）。
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

RES = Path("results")
FIG = Path("results/figs")
FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.load(open(RES / name, encoding="utf-8"))


def domain_figs(prefix, label):
    fa = RES / f"{prefix}_A.json"
    if not fa.exists():
        print(f"[skip] {prefix}: 无数据")
        return None
    a = load(f"{prefix}_A.json")
    b = load(f"{prefix}_B.json")
    c = load(f"{prefix}_C.json")

    groups = {"A": a, "B": b, "C": c}
    dims = ["factual", "completeness", "logic", "depth"]

    # 公平对比：三组都非退化(>=30字)的共同题
    def qset(d):
        return set(r["question"] for r in d if len(r["eval_answer"]) >= 30)
    common = qset(a) & qset(b) & qset(c)
    idx = {g: {r["question"]: r for r in d} for g, d in groups.items()}

    stats = {}
    for g in "ABC":
        recs_all = groups[g]
        recs_com = [idx[g][q] for q in common]
        Y_all = [r["Y"] for r in recs_all if r["Y"] is not None]
        Y_com = [r["Y"] for r in recs_com if r["Y"] is not None]
        M = [r["M"] for r in recs_all if r.get("M")]
        dim_means = {}
        for dm in dims:
            vals = []
            for q in common:
                pj = idx[g][q]["Y_detail"]["per_judge"]
                v = [jv[dm] for jv in pj.values() if jv and dm in jv]
                if v:
                    vals.append(st.mean(v))
            dim_means[dm] = st.mean(vals) if vals else 0
        stats[g] = {
            "Y_all": Y_all, "Y_com": Y_com, "M": M,
            "meanY_all": st.mean(Y_all), "meanY_com": st.mean(Y_com) if Y_com else 0,
            "meanM": st.mean(M) if M else 0, "dims": dim_means,
            "n_com": len(common),
        }

    colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#C44E52"}

    # 图1：三组 Y 均值柱状（全量 vs 公平对比）
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = range(3)
    w = 0.36
    all_v = [stats[g]["meanY_all"] for g in "ABC"]
    com_v = [stats[g]["meanY_com"] for g in "ABC"]
    ax.bar([i - w / 2 for i in x], all_v, w, label="全部100条", color="#8DA0CB")
    ax.bar([i + w / 2 for i in x], com_v, w, label=f"剔除退化({stats['A']['n_com']}条)", color="#E78AC3")
    for i, v in enumerate(all_v):
        ax.text(i - w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(com_v):
        ax.text(i + w / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["A (X=0,无反问)", "B (X=1,普通反问)", "C (X=2,黄金反问)"])
    ax.set_ylabel("回答质量 Y (1-10)")
    ax.set_title(f"{label}：三组回答质量对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_Y_bar.png", dpi=150)
    plt.close(fig)

    # 图2：四维雷达式分组柱（公平对比题）
    fig, ax = plt.subplots(figsize=(7, 4.2))
    nd = len(dims)
    bw = 0.25
    for gi, g in enumerate("ABC"):
        vals = [stats[g]["dims"][d] for d in dims]
        ax.bar([i + gi * bw for i in range(nd)], vals, bw,
               label=f"{g}组", color=colors[g])
    dim_cn = {"factual": "事实性", "completeness": "完备度",
              "logic": "逻辑条理", "depth": "分析深度"}
    ax.set_xticks([i + bw for i in range(nd)])
    ax.set_xticklabels([dim_cn[d] for d in dims])
    ax.set_ylabel("维度得分 (1-10)")
    ax.set_title(f"{label}：四维质量分解（剔除退化，n={stats['A']['n_com']}）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_dims.png", dpi=150)
    plt.close(fig)

    # 图3：Y 分布箱线
    fig, ax = plt.subplots(figsize=(6, 4.2))
    data = [stats[g]["Y_all"] for g in "ABC"]
    bp = ax.boxplot(data, tick_labels=["A", "B", "C"], patch_artist=True, showmeans=True)
    for patch, g in zip(bp["boxes"], "ABC"):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.6)
    ax.set_ylabel("回答质量 Y (1-10)")
    ax.set_title(f"{label}：回答质量分布")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_box.png", dpi=150)
    plt.close(fig)

    print(f"[ok] {label}: 生成 3 图 -> {FIG}")
    return stats


def main():
    poetry = domain_figs("poetry", "古诗词")
    recipe = domain_figs("recipe", "菜谱")

    # 跨领域对照图（若菜谱已就绪）
    if poetry and recipe:
        # 图1：两领域 factual 与 Y 随 X 的绝对走势
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, stats, label in [(axes[0], poetry, "古诗词"), (axes[1], recipe, "菜谱")]:
            xs = [0, 1, 2]
            fac = [stats[g]["dims"]["factual"] for g in "ABC"]
            y = [stats[g]["meanY_all"] for g in "ABC"]
            ax.plot(xs, fac, "o-", color="#C44E52", label="事实性")
            ax.plot(xs, y, "s--", color="#4C72B0", label="回答质量Y")
            ax.set_xticks(xs)
            ax.set_xticklabels(["A(X=0)", "B(X=1)", "C(X=2)"])
            ax.set_xlabel("反问强度 X")
            ax.set_ylabel("得分 (1-10)")
            ax.set_title(label)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("跨领域对照：反问强度对事实性与回答质量的影响", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG / "cross_domain.png", dpi=150)
        plt.close(fig)
        print(f"[ok] 跨领域对照图 -> {FIG / 'cross_domain.png'}")

        # 图2：领域调节判决图 —— 相对A的变化量(ΔY, Δfactual)，最能体现"方向相反"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        metrics = [("meanY_all", "回答质量 ΔY", axes[0]),
                   ("factual", "事实性 Δfactual", axes[1])]
        for key, title, ax in metrics:
            for stats, label, color in [(poetry, "古诗词(主观)", "#C44E52"),
                                        (recipe, "菜谱(客观)", "#4C72B0")]:
                if key == "factual":
                    base = stats["A"]["dims"]["factual"]
                    vals = [stats[g]["dims"]["factual"] - base for g in "ABC"]
                else:
                    base = stats["A"][key]
                    vals = [stats[g][key] - base for g in "ABC"]
                ax.plot([0, 1, 2], vals, "o-", color=color, label=label, linewidth=2)
            ax.axhline(0, color="gray", ls=":", alpha=0.6)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["A", "B", "C"])
            ax.set_xlabel("反问强度 X")
            ax.set_ylabel(title + " (相对A)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("领域调节判决图：反问对两领域的差异化影响（相对无反问基线）", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG / "domain_modulation.png", dpi=150)
        plt.close(fig)
        print(f"[ok] 领域调节判决图 -> {FIG / 'domain_modulation.png'}")

    # 输出汇总数字
    summary = {}
    for prefix, stats in [("poetry", poetry), ("recipe", recipe)]:
        if not stats:
            continue
        summary[prefix] = {
            g: {"meanY_all": round(stats[g]["meanY_all"], 3),
                "meanY_com": round(stats[g]["meanY_com"], 3),
                "meanM": round(stats[g]["meanM"], 3),
                "dims": {k: round(v, 3) for k, v in stats[g]["dims"].items()}}
            for g in "ABC"}
    json.dump(summary, open(RES / "figs_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[ok] 汇总数字 -> results/figs_summary.json")


if __name__ == "__main__":
    main()
