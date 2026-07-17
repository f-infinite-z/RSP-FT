"""
M-Y边际递减图（0.5B/1.5B/3B散点+回归）。来源：matplotlib+Pearson
输出: results/paper_figs/fig_my_scaling.png
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, statistics as st, math
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
RES = Path("results")

def load(fn): return json.load(open(RES/fn, encoding="utf-8"))
def pearson(xs, ys):
    mx,my=st.mean(xs),st.mean(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs));sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else 0.0

scales = [("0.5B","recipe_A.json","recipe_B.json"),
          ("1.5B","recipe_1.5b_A.json","recipe_1.5b_B.json"),
          ("3B","recipe_3b_A.json","recipe_3b_B.json")]

data=[]
for name,fa,fb in scales:
    A=load(fa); B=load(fb)
    a_y=st.mean([r["Y"] for r in A])
    M=[r["M"] for r in B if r.get("M") and r["M"]>0]
    Y=[r["Y"] for r in B if r.get("M") and r["M"]>0]
    # 也并入C组(如有)提高散点密度——菜谱有C
    fc=fa.replace("_A","_C").replace("recipe_A","recipe_C")
    try:
        C=load(fc)
        for r in C:
            if r.get("M") and r["M"]>0: M.append(r["M"]);Y.append(r["Y"])
    except: pass
    data.append({"name":name,"a_y":a_y,"M":M,"Y":Y,"r":pearson(M,Y)})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ===图A: 双轴 r柱+基线折线===
ax1=axes[0]
names=[d["name"] for d in data]; rs=[d["r"] for d in data]; ays=[d["a_y"] for d in data]
x=range(len(names))
bars=ax1.bar(x, rs, width=0.5, color="#4C72B0", alpha=0.85, label="连续 r(M,Y)")
ax1.set_ylabel("连续 r(M, Y)", color="#4C72B0", fontsize=12)
ax1.set_ylim(0, 0.55)
ax1.set_xticks(list(x)); ax1.set_xticklabels(names, fontsize=12)
ax1.set_xlabel("模型规模", fontsize=12)
for i,r in enumerate(rs): ax1.text(i, r+0.012, "%.3f"%r, ha="center", fontweight="bold", color="#4C72B0")
ax2=ax1.twinx()
ax2.plot(x, ays, "o-", color="#C44E52", lw=2.2, ms=9, label="A组(无反问)基线 Y")
ax2.set_ylabel("A 组基线回答质量 Y", color="#C44E52", fontsize=12)
ax2.set_ylim(3, 5.6)
for i,y in enumerate(ays): ax2.text(i, y+0.08, "%.2f"%y, ha="center", color="#C44E52", fontweight="bold")
ax1.set_title("(a) M–Y 相关随规模递减 vs 基线随规模升高\n（边际增益递减：基线越高，反问可填补的缺口越小）", fontsize=11.5)
l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper center", fontsize=10)

# ===图B: 三规模散点+回归===
ax=axes[1]
colors=["#4C72B0","#55A868","#C44E52"]
for d,c in zip(data,colors):
    M,Y=d["M"],d["Y"]
    ax.scatter(M,Y,s=14,alpha=0.35,color=c)
    # 回归线
    mx,my=st.mean(M),st.mean(Y)
    b=sum((m-mx)*(y-my) for m,y in zip(M,Y))/sum((m-mx)**2 for m in M)
    xs=[min(M),max(M)]; ys=[my+b*(xx-mx) for xx in xs]
    ax.plot(xs,ys,color=c,lw=2.5,label="%s (r=%.2f)"%(d["name"],d["r"]))
ax.set_xlabel("反问质量 M", fontsize=12); ax.set_ylabel("回答质量 Y", fontsize=12)
ax.set_title("(b) 三规模 M–Y 散点与回归\n（斜率随规模变缓，方向始终为正）", fontsize=11.5)
ax.legend(fontsize=10, title="模型规模")

fig.tight_layout()
out=RES/"paper_figs"; out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/"fig_my_scaling.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved:", out/"fig_my_scaling.png")
for d in data: print("  %s: r=%.3f 基线Y=%.2f n=%d"%(d["name"],d["r"],d["a_y"],len(d["M"])))
