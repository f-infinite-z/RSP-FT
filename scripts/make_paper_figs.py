# -*- coding: utf-8 -*-
"""论文正式图 v2：真实 GC（indep_gn）版
图2 loss/GN 诊断曲线（2×2） / 图3A GC 迁移（2×2） / 图3B 冲突比例 / 图4 干预Y
+ fig_excel_data.csv（人工复现数据导出）
"""
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"<paper-data-location>\results")
OUT = Path(r"<paper-data-location>\project\results\paper_figs")
OUT.mkdir(parents=True, exist_ok=True)
DIAG = Path(r"<paper-data-location>\project\results\grad_diag")
EXCEL = Path(r"<paper-data-location>\project\results\paper_figs\fig_excel_data")
EXCEL.mkdir(parents=True, exist_ok=True)


def load(f):
    r = json.load(open(ROOT / (f + ".json"), encoding="utf-8"))
    return [x["Y"] for x in r if x.get("Y") is not None]


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False)


# ---------- 真实 GC（indep_gn：段求和 → step 归一化 → iter 平均） ----------
def gc_and_loss(csv_name, segs=("ans_first", "cq", "ans_final")):
    df = pd.read_csv(DIAG / csv_name, encoding="utf-8")
    df["indep_gn"] = df["indep_gn"].astype(float)
    grp = df.groupby(["iter", "step", "seg"])["indep_gn"].sum().reset_index()
    tot = grp.groupby(["iter", "step"])["indep_gn"].transform("sum")
    grp["gc"] = grp["indep_gn"] / tot
    gc = {}
    for it, g in grp.groupby("iter"):
        gc[int(it)] = {s: float(g.loc[g["seg"] == s, "gc"].mean()) for s in segs}
    loss = {}
    for it, g in df.groupby("iter"):
        loss[int(it)] = {s: float(g.loc[g["seg"] == s, "loss"].mean()) for s in segs}
    gn = {}
    for it, g in df.groupby("iter"):
        gn[int(it)] = {s: float(g.loc[g["seg"] == s, "gn_cum"].mean()) for s in segs}
    return gc, loss, gn


SEGS = ("ans_first", "cq", "ans_final")
LABELS = {"ans_first": "初答", "cq": "反问", "ans_final": "终答"}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
GROUPS = [("B", "B_v2.csv", "无调度"), ("B_plus", "B_rqls_diag.csv", "CQLS"),
          ("C", "C_v2.csv", "无调度"), ("C_plus", "C_rqls_diag.csv", "CQLS")]

# 预计算
DATA = {}
for gname, csvn, tag in GROUPS:
    DATA[gname] = gc_and_loss(csvn)


# ---------- 图2：三段 loss + 累积 GN（2×2：行=B/C，列=无调度/CQLS） ----------
def plot_2x2(metric, ylabel, figname, title):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    for i, base in enumerate(["B", "C"]):
        for j, suffix in enumerate(["", "_plus"]):
            ax = axes[i][j]
            gname = base + suffix
            d = DATA[gname][{"loss": 1, "gn": 2}[metric]]
            iters = sorted(d.keys())
            for k, s in enumerate(SEGS):
                ax.plot(iters, [d[it][s] for it in iters], "o-", label=LABELS[s],
                        color=COLORS[k], linewidth=1.6, markersize=5)
            ax.set_title(f"组 {base}{'（CQLS）' if suffix else '（无调度）'}", fontsize=10)
            ax.set_xticks(iters)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("训练迭代")
            ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    axes[0][0].legend(fontsize=9, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / figname, dpi=150)
    plt.close(fig)


plot_2x2("loss", "段损失（未加权）", "fig2_loss_curves.png", "图2a 三段损失曲线")
plot_2x2("gn", "累积梯度范数", "fig2_gn_curves.png", "图2b 三段累积梯度范数")


# ---------- 图3A：真实 GC 迁移（2×2） ----------
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True, sharey=True)
for i, base in enumerate(["B", "C"]):
    for j, suffix in enumerate(["", "_plus"]):
        ax = axes[i][j]
        gname = base + suffix
        gc = DATA[gname][0]
        iters = sorted(gc.keys())
        for k, s in enumerate(SEGS):
            ax.plot(iters, [gc[it][s] for it in iters], "o-", label=LABELS[s],
                    color=COLORS[k], linewidth=1.6, markersize=5)
        ax.set_title(f"组 {base}{'（CQLS）' if suffix else '（无调度）'}", fontsize=10)
        ax.set_xticks(iters)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("梯度贡献率 GC")
        if i == 1:
            ax.set_xlabel("训练迭代")
        ax.grid(alpha=0.3)
fig.suptitle("图3A 梯度贡献率迁移：无调度 vs CQLS（真实 GC）", fontsize=12)
axes[0][0].legend(fontsize=9, loc="upper right")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "fig3a_gc_migration.png", dpi=150)
plt.close(fig)


# ---------- 图3B：冲突比例 ----------
fig, ax = plt.subplots(figsize=(4.5, 3.6))
rates = [41, 64]
bars = ax.bar(["B 组", "C 组"], rates, color=["#4C72B0", "#DD8452"], width=0.5)
ax.set_ylabel("段间梯度负相关占比 (%)")
ax.set_title("图3B 梯度方向冲突比例", fontsize=11)
ax.set_ylim(0, 80)
for b, v in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, "%d%%" % v, ha="center", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig3b_conflict_ratio.png", dpi=150)
plt.close(fig)


# ---------- 图4：干预 × 控长度 Y ----------
strategies = ["无调度", "CQLS", "mu2", "mu6", "PCGrad"]
raw_map = {
    "B": ["rqls_B", "rqls_B_rqls", "cqls_plus_B", "cqls_strong_B", "pcgrad_B"],
    "C": ["rqls_C", "rqls_C_rqls", "cqls_plus_C", "cqls_strong_C", "pcgrad_C"],
}
cnc_map = {
    "B": ["concise_B", "concise_B_cqls", "concise_B_mu2", "concise_B_mu6", "concise_B_pc"],
    "C": ["concise_C", "concise_C_cqls", "concise_C_mu2", "concise_C_mu6", "concise_C_pc"],
}
A = load("rqls_A")
a_mean = sum(A) / len(A)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for gi, grp in enumerate(["B", "C"]):
    ax = axes[gi]
    raw_m = [sum(load(f)) / len(load(f)) for f in raw_map[grp]]
    cnc_m = [sum(load(f)) / len(load(f)) for f in cnc_map[grp]]
    x = np.arange(len(strategies))
    w = 0.36
    ax.bar(x - w / 2, raw_m, w, label="原始", color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, cnc_m, w, label="+控长度", color="#DD8452", alpha=0.85)
    ax.axhline(a_mean, color="gray", linestyle="--", linewidth=1)
    ax.text(len(strategies) - 0.4, a_mean + 0.03, "A基线 %.2f" % a_mean, fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(["无调度", "CQLS", "μ2", "μ6", "PCGrad"], fontsize=9)
    ax.set_title("组 %s" % grp, fontsize=11)
    if gi == 0:
        ax.set_ylabel("回答质量 Y")
    for i in range(len(strategies)):
        t, p = welch(load(cnc_map[grp][i]), A)
        if p < 0.05:
            ax.text(i + w / 2, cnc_m[i] + 0.08, "*", fontsize=11, ha="center")
        t2, p2 = welch(load(cnc_map[grp][i]), load(raw_map[grp][i]))
        if p2 < 0.05:
            ax.text(i + w / 2, cnc_m[i] + 0.22, "+", fontsize=10, ha="center", color="#C44E52")
ax.legend(fontsize=8, loc="lower left")
fig.suptitle("图4 干预策略与控长度对回答质量的影响（*: 与A基线 p<0.05；+: 控长度显著）", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig4_intervention_Y.png", dpi=150)
plt.close(fig)


# ---------- Excel 数据导出（人工复现用） ----------
def dump(name, header, rows):
    p = EXCEL / (name + ".csv")
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[excel] {p.name} ({len(rows)} 行)")


# 图2/3A 数据：每组的 loss/gn/gc 逐 iter
for gname, csvn, tag in GROUPS:
    gc, loss, gn = DATA[gname]
    iters = sorted(gc.keys())
    rows = []
    for it in iters:
        rows.append([it] + [round(loss[it][s], 4) for s in SEGS]
                    + [round(gn[it][s], 3) for s in SEGS]
                    + [round(gc[it][s], 4) for s in SEGS])
    dump(f"diag_{gname}", ["iter"] + [f"L_{s}" for s in SEGS]
         + [f"GN_{s}" for s in SEGS] + [f"GC_{s}" for s in SEGS], rows)

# 图4 数据
rows = []
for grp in ["B", "C"]:
    for i, st in enumerate(strategies):
        r = load(raw_map[grp][i]); c = load(cnc_map[grp][i])
        rows.append([grp, st, round(sum(r) / len(r), 4), round(sum(c) / len(c), 4)])
dump("intervention_Y", ["组", "策略", "原始Y", "控长度Y"], rows)

# 图3B 数据
dump("conflict_ratio", ["组", "负相关占比%"], [["B", 41], ["C", 64]])

print("\n全部完成，输出目录:", OUT)
