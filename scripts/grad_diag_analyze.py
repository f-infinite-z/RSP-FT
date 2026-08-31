"""
梯度诊断分析：从 SegmentDiag CSV 生成段损失曲线 + 累积梯度范数曲线 + GC 贡献率报告
支持任意段名（三段反问结构 / 两段多任务结构）
用法：python -X utf8 grad_diag_analyze.py --diag results/grad_diag/B_diag.csv --out results/grad_diag/B [--seg-names ans_first,cq,ans_final]
"""
import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def analyze(diag_csv: Path, out_dir: Path, seg_names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(diag_csv, encoding="utf-8")
    segs_all = sorted(set(df["seg"])) if "seg" in df.columns else []
    SEG = seg_names if seg_names else ([s for s in ["ans_first", "cq", "ans_final"] if s in segs_all] or segs_all)

    wide = df.pivot_table(index=["iter", "step"], columns="seg",
                          values=["loss", "gn_cum"], aggfunc="first").reset_index()
    wide.columns = ["iter", "step"] + [f"{a}_{b}" for a, b in wide.columns[2:]]
    for s in SEG:
        if f"loss_{s}" not in wide.columns:
            wide[f"loss_{s}"] = float("nan")
            wide[f"gn_cum_{s}"] = float("nan")
    wide = wide.sort_values(["iter", "step"]).reset_index(drop=True)

    n = max(1, len(wide) // 100)
    sm = wide.rolling(n, min_periods=1).mean()

    # ---- 图1：段损失曲线 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, c in zip(SEG, COLORS):
        ax.plot(sm.index, sm[f"loss_{s}"], label=s, color=c, linewidth=1.6)
    for it in wide["iter"].unique()[1:]:
        idx = wide.index[wide["iter"] == it][0]
        ax.axvline(idx, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("训练 step（竖虚线=iter 边界）")
    ax.set_ylabel("段损失（未加权）")
    ax.set_title("段损失曲线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ---- 图2：累积梯度范数曲线 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, c in zip(SEG, COLORS):
        ax.plot(sm.index, sm[f"gn_cum_{s}"], label=s + "（累积）", color=c, linewidth=1.6)
    for it in wide["iter"].unique()[1:]:
        idx = wide.index[wide["iter"] == it][0]
        ax.axvline(idx, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("训练 step（竖虚线=iter 边界）")
    ax.set_ylabel("累积梯度范数（LoRA 参数）")
    ax.set_title("累积梯度范数：加入各段后的增量反映其梯度贡献")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norm.png", dpi=150)
    plt.close(fig)

    # ---- 报告 ----
    lines = []
    add = lines.append
    add(f"梯度诊断汇总: {diag_csv.name}")
    add(f"段: {SEG}  样本数={len(wide)} 平滑窗口={n}")
    add("\n【1】各 iter 平均段损失")
    hdr = f"{'iter':>4}" + "".join(f" {('L_'+s):>14}" for s in SEG)
    add(hdr)
    for it, g in wide.groupby("iter"):
        row = f"{int(it):>4}" + "".join(f" {g[f'loss_{s}'].mean():>14.4f}" for s in SEG)
        add(row)

    add("\n【2】各 iter 平均累积梯度范数")
    hdr = f"{'iter':>4}" + "".join(f" {('gn_'+s):>12}" for s in SEG)
    add(hdr)
    for it, g in wide.groupby("iter"):
        row = f"{int(it):>4}" + "".join(f" {g[f'gn_cum_{s}'].mean():>12.4f}" for s in SEG)
        add(row)

    add("\n【3】梯度贡献率 GC（真实值：段独立梯度范数占比；旧数据 fallback 增量近似）")
    has_indep = "indep_gn" in df.columns
    if has_indep:
        gc_df = df.copy()
        gc_df["indep_gn"] = gc_df["indep_gn"].astype(float)
        # 段级贡献 = (iter,step,seg) 组内各样本独立范数之和 → step 内归一化 → (iter,seg) 平均
        grp = gc_df.groupby(["iter", "step", "seg"])["indep_gn"].sum().reset_index()
        tot = grp.groupby(["iter", "step"])["indep_gn"].transform("sum")
        grp["gc"] = grp["indep_gn"] / tot
        hdr = f"{'iter':>4}" + "".join(f" {('GC_'+s):>12}" for s in SEG)
        add(hdr)
        for it, g in grp.groupby("iter"):
            row = f"{int(it):>4}"
            for s in SEG:
                v = g.loc[g["seg"] == s, "gc"]
                row += f" {v.mean():>12.3f}" if len(v) else f" {'nan':>12}"
            add(row)
        add("（真实 GC：段独立梯度范数 / 该 step 各段独立范数之和（step 内归一化），再按 iter 平均；"
            "与旧增量近似不一致时以本表为准）")
    else:
        gcols = []
        prev = None
        for s in SEG:
            col = f"g_{s}"
            if prev is None:
                wide[col] = wide[f"gn_cum_{s}"].clip(lower=0)
            else:
                wide[col] = (wide[f"gn_cum_{s}"] - wide[prev]).clip(lower=0)
            gcols.append(col)
            prev = f"gn_cum_{s}"
        total_g = sum(wide[c] for c in gcols)
        for c in gcols:
            wide[c] = wide[c] / total_g
        hdr = f"{'iter':>4}" + "".join(f" {('GC_'+s):>12}" for s in SEG)
        add(hdr)
        for it, idx in wide.groupby("iter").groups.items():
            row = f"{int(it):>4}" + "".join(f" {wide[c].loc[idx].mean():>12.3f}" for c in gcols)
            add(row)
        add("（fallback：增量近似 + 累计均值；注意 batch 内累积与段间抵消会低估后期段贡献）")

    add("\n【4】判读")
    add(" - 某段 GC 占比异常高(>50%) → 该段喧宾夺主；占比过低(<10%) → 被淹没")
    add(" - 若加入某段后累积范数几乎无增量 → 该段梯度贡献弱（或被方向冲突抵消）")

    txt = "\n".join(lines)
    (out_dir / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n图与报告已保存: {out_dir}")
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True, help="SegmentDiag CSV 路径")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--seg-names", default="", help="逗号分隔的段名（默认自动识别）")
    args = ap.parse_args()
    seg = args.seg_names.split(",") if args.seg_names else None
    analyze(Path(args.diag), Path(args.out), seg)


if __name__ == "__main__":
    main()
