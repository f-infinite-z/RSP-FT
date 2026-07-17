"""
RSP-FT方法总览图（A/B/C流程+X→M→Y中介框架）。来源：matplotlib+FancyBboxPatch
输出：results/paper_figs/fig_pipeline.png (300dpi)
用法：python make_pipeline_fig.py
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_xlim(0, 13); ax.set_ylim(-1.7, 9); ax.axis("off")

# 配色
C_A = "#8ECae6"; C_B = "#FFB703"; C_C = "#FB8500"
C_BOX = "#F1F3F5"; C_LOSS = "#E9ECEF"; C_MED = "#D8F3DC"
EDGE = "#495057"

def box(x, y, w, h, text, fc, fs=10, bold=False, ec=EDGE):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                       fc=fc, ec=ec, lw=1.3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, color=color, lw=lw, linestyle=ls))

# 标题
ax.text(6.5, 8.6, "反问式自博弈微调（RSP-FT）：三组分层操纵与链式中介",
        ha="center", fontsize=14, fontweight="bold")

# ===== 三列表头 =====
cols = [(1.0, C_A, "A 组（X=0）\n单向问答·基线"),
        (5.0, C_B, "B 组（X=1）\n往复对话·普通反问"),
        (9.0, C_C, "C 组（X=2）\n往复对话·黄金反问")]
for x, c, t in cols:
    box(x, 7.4, 3.0, 0.75, t, c, fs=10.5, bold=True)

# ===== A 组流程 =====
box(1.0, 6.2, 3.0, 0.7, "问题 q", C_BOX, 10)
arrow(2.5, 6.2, 2.5, 5.75)
box(1.0, 5.0, 3.0, 0.7, "M_B 生成回答 ans", C_BOX, 10)
arrow(2.5, 5.0, 2.5, 4.4)
box(1.0, 3.5, 3.0, 0.85, "损失\nL_A = CE(ans|q)", C_LOSS, 9.5)

# ===== B 组流程 =====
by = [("问题 q", C_BOX), ("M_B 初答 ans1 + 普通反问 cq_low", C_BOX),
      ("M_A 仿真补充 r", "#FFE8CC"), ("M_B 最终回答 ans2", C_BOX)]
yy = 6.2
for i, (t, c) in enumerate(by):
    h = 0.7 if i != 1 else 0.8
    box(5.0, yy, 3.0, h, t, c, 9.3)
    if i < len(by) - 1:
        arrow(6.5, yy, 6.5, yy - 0.45)
    yy -= 1.15
box(5.0, 1.9, 3.0, 1.0, "损失 L_B =\nCE(ans1|q)\n+ CE(cq_low|q,ans1)\n+ CE(ans2|q,ans1,cq_low,r)", C_LOSS, 7.2)
arrow(6.5, 2.9, 6.5, 2.9)

# ===== C 组流程 =====
cy = [("问题 q", C_BOX), ("M_B 初答 ans1 + 黄金反问 cq_high", "#FFD6A5"),
      ("M_A 仿真补充 r", "#FFE8CC"), ("M_B 最终回答 ans2", C_BOX)]
yy = 6.2
for i, (t, c) in enumerate(cy):
    h = 0.7 if i != 1 else 0.8
    box(9.0, yy, 3.0, h, t, c, 9.3)
    if i < len(cy) - 1:
        arrow(10.5, yy, 10.5, yy - 0.45)
    yy -= 1.15
box(9.0, 1.9, 3.0, 1.0, "损失 L_C =\nCE(ans1|q)\n+ CE(cq_high|q,ans1)\n+ CE(ans2|q,ans1,cq_high,r)", C_LOSS, 7.2)
arrow(10.5, 2.9, 10.5, 2.9)

# B/C 唯一差异标注
ax.annotate("唯一差异：\n反问质量", xy=(8.0, 5.4), xytext=(8.0, 5.4),
            ha="center", va="center", fontsize=9, color="#C1121F", fontweight="bold")
arrow(7.35, 5.4, 6.9, 5.4, color="#C1121F", lw=1.2)
arrow(8.65, 5.4, 9.1, 5.4, color="#C1121F", lw=1.2)

# ===== 下方中介框 =====
box(0.6, 0.35, 11.8, 1.0, "", C_MED, 10)
ax.text(1.2, 1.05, "链式中介分析", fontsize=10.5, fontweight="bold", va="center")
mx = [(3.3, "X\n分组强度 0/1/2"), (6.4, "M\n反问质量(裁判)"), (9.8, "Y\n回答质量(裁判)")]
for x, t in mx:
    box(x, 0.5, 1.7, 0.7, t, "#FFFFFF", 9)
arrow(5.0, 0.85, 6.4, 0.85, color="#2D6A4F", lw=1.6)
arrow(8.1, 0.85, 9.8, 0.85, color="#2D6A4F", lw=1.6)
ax.text(5.7, 1.05, "a", fontsize=9, color="#2D6A4F", fontweight="bold")
ax.text(8.95, 1.05, "b", fontsize=9, color="#2D6A4F", fontweight="bold")

# 评测口径注
ax.text(6.5, 0.15, "所有组统一评判 M_B 最终回答（A 组即其唯一回答），保证因变量口径一致",
        ha="center", fontsize=8, color="#666", style="italic")

# ===== 底部符号注释 =====
note1 = "符号说明：CE(x|context)=给定上下文 context 生成文本 x 的交叉熵损失；  q=问题；  ans=A组唯一回答；  ans1=首轮初答；"
note2 = "cq_low / cq_high=普通 / 高质量（黄金）反问；  r=提问者 M_A 的仿真补充；  ans2=融合反问反馈后的最终回答。"
box(0.6, -1.5, 11.8, 1.15, "", "#FFF9DB", 8)
ax.text(6.5, -0.72, note1, fontsize=7.6, va="center", ha="center", color="#333")
ax.text(6.5, -1.18, note2, fontsize=7.6, va="center", ha="center", color="#333")

out = Path("results/paper_figs"); out.mkdir(parents=True, exist_ok=True)
fp = out / "fig_pipeline.png"
fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
print("saved:", fp)

