"""
论文核心四图（M-Y散点/训练曲线/长度混淆/四维双刃剑）。来源：matplotlib+Pearson回归

用法：python make_paper_figs.py
输出：results/paper_figs/*.png
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

RES = Path("results")
FIG = Path("results/paper_figs")
FIG.mkdir(parents=True, exist_ok=True)
DIMS = ["factual", "completeness", "logic", "depth"]
DIM_CN = {"factual": "事实性", "completeness": "完备度", "logic": "逻辑", "depth": "深度"}


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


def collect_my(prefix):
    my = []
    for g in "BC":
        d = load(f"{prefix}_{g}")
        if not d:
            continue
        for r in d:
            if r.get("M") and r["Y"] is not None:
                my.append((float(r["M"]), float(r["Y"])))
    return my


# ---- 图1：M-Y 散点 + 回归线（核心发现）----
def fig1_my_scatter():
    datasets = [("MVE菜谱", "recipe", "#4C72B0"),
                ("lr5菜谱", "recipe_lr5", "#DD8452"),
                ("古诗词", "poetry", "#55A868")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (label, prefix, color) in zip(axes, datasets):
        my = collect_my(prefix)
        if not my:
            continue
        xs = [p[0] for p in my]
        ys = [p[1] for p in my]
        r = pearson(xs, ys)
        ax.scatter(xs, ys, s=14, alpha=0.4, color=color)
        # 回归线
        n = len(xs)
        mx, myv = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - myv) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        b = myv - slope * mx
        xr = [min(xs), max(xs)]
        ax.plot(xr, [slope * x + b for x in xr], color="red", lw=2)
        ax.set_title(f"{label}  r={r:+.3f}")
        ax.set_xlabel("反问质量 M")
        ax.set_ylabel("回答质量 Y")
        ax.grid(alpha=0.3)
    fig.suptitle("核心发现：反问质量 M 正向决定回答质量 Y（三数据集一致）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_MY_scatter.png", dpi=150)
    plt.close(fig)
    print("[ok] fig1_MY_scatter")


# ---- 图2：训练强度(lr) vs M-Y相关 + Y均值 ----
def fig2_lr_curve():
    points = [("1e-6", "recipe", 1e-6), ("2e-6", "recipe_lr2e6", 2e-6),
              ("5e-6", "recipe_lr5", 5e-6), ("1e-5", "recipe_lr1e5", 1e-5)]
    lrs, rs, ymeans = [], [], []
    for label, prefix, lrval in points:
        my = collect_my(prefix)
        if not my:
            continue
        xs = [p[0] for p in my]
        ys = [p[1] for p in my]
        lrs.append(label)
        rs.append(pearson(xs, ys))
        ymeans.append(st.mean(ys))
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = range(len(lrs))
    ax1.plot(x, rs, "o-", color="#C44E52", lw=2, label="M-Y相关 r")
    ax1.set_xlabel("自博弈学习率 (训练强度→)")
    ax1.set_ylabel("M-Y 相关系数 r", color="#C44E52")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(lrs)
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax1.grid(alpha=0.3)
    for xi, ri in zip(x, rs):
        ax1.text(xi, ri + 0.01, f"{ri:.3f}", ha="center", fontsize=9, color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(x, ymeans, "s--", color="#4C72B0", lw=2, label="Y均值")
    ax2.set_ylabel("回答质量 Y 均值", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    fig.suptitle("训练强度调节：训练越强→M-Y相关越弱+Y越低（主任务挤占）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_lr_curve.png", dpi=150)
    plt.close(fig)
    print("[ok] fig2_lr_curve")


# ---- 图3：长度混淆 ----
def fig3_length():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左：各组长度（MVE菜谱）
    groups = {g: load(f"recipe_{g}") for g in "ABC"}
    if all(groups.values()):
        data = [[len(r["eval_answer"]) for r in groups[g]] for g in "ABC"]
        bp = ax1.boxplot(data, tick_labels=["A", "B", "C"], patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#4C72B0", "#DD8452", "#C44E52"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax1.set_ylabel("答案长度（字）")
        ax1.set_title("反问强度↑ → 答案变长")
        ax1.grid(axis="y", alpha=0.3)
    # 右：长度 vs Y 散点（全体）
    allL, allY = [], []
    for g in "ABC":
        for r in groups[g]:
            if r["Y"] is not None:
                allL.append(len(r["eval_answer"]))
                allY.append(r["Y"])
    r = pearson(allL, allY)
    ax2.scatter(allL, allY, s=14, alpha=0.4, color="#8172B3")
    n = len(allL)
    mx, my = sum(allL) / n, sum(allY) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(allL, allY)) / sum((x - mx) ** 2 for x in allL)
    b = my - slope * mx
    xr = [min(allL), max(allL)]
    ax2.plot(xr, [slope * x + b for x in xr], color="red", lw=2)
    ax2.set_xlabel("答案长度（字）")
    ax2.set_ylabel("回答质量 Y")
    ax2.set_title(f"长度 vs Y  r={r:+.3f}（长答案被惩罚）")
    ax2.grid(alpha=0.3)
    fig.suptitle("长度膨胀混淆：反问使答案变长，而长答案 Y 更低", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_length.png", dpi=150)
    plt.close(fig)
    print("[ok] fig3_length")


# ---- 图4：四维分解（双刃剑）----
def fig4_dims():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (label, prefix) in zip(axes, [("古诗词", "poetry"), ("菜谱", "recipe")]):
        groups = {g: load(f"{prefix}_{g}") for g in "ABC"}
        if not all(groups.values()):
            continue

        def dim_mean(recs, dm):
            vals = []
            for r in recs:
                pj = r["Y_detail"]["per_judge"]
                v = [jv[dm] for jv in pj.values() if jv and dm in jv]
                if v:
                    vals.append(st.mean(v))
            return st.mean(vals) if vals else 0
        nd = len(DIMS)
        bw = 0.25
        for gi, g in enumerate("ABC"):
            vals = [dim_mean(groups[g], dm) for dm in DIMS]
            ax.bar([i + gi * bw for i in range(nd)], vals, bw, label=f"{g}组",
                   color=["#4C72B0", "#DD8452", "#C44E52"][gi])
        ax.set_xticks([i + bw for i in range(nd)])
        ax.set_xticklabels([DIM_CN[d] for d in DIMS])
        ax.set_ylabel("维度得分")
        ax.set_title(label)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("四维分解：高强度反问(C)的双刃剑（深度↑ 事实性↓）", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_dims.png", dpi=150)
    plt.close(fig)
    print("[ok] fig4_dims")


def main():
    fig1_my_scatter()
    fig2_lr_curve()
    fig3_length()
    fig4_dims()
    print(f"\n全部论文图已保存至 {FIG}")


if __name__ == "__main__":
    main()
