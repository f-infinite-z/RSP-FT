"""
三领域对照图（领域A/B Y+r(M,Y)+C双刃剑）。来源：matplotlib
输出: results/paper_figs/fig_domain.png
说明：r 不随事实约束单调（菜谱0.48>诗词0.36>法律0.15），法律低因能力边际递减
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, statistics as st, math
from pathlib import Path
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False
RES=Path("results")
def load(fn): 
    p=RES/fn
    return json.load(open(p,encoding="utf-8")) if p.exists() else None
def pearson(xs,ys):
    mx,my=st.mean(xs),st.mean(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs));sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else 0.0
def dim_mean(recs,dim):
    v=[]
    for r in recs:
        pj=r.get("Y_detail",{}).get("per_judge",{})
        ds=[x[dim] for x in pj.values() if x.get(dim) is not None]
        if ds: v.append(st.mean(ds))
    return st.mean(v) if v else None

doms=[("古诗词","poetry"),("菜谱","recipe"),("法律","law")]
D=[]
for name,pre in doms:
    A=load("%s_A.json"%pre);B=load("%s_B.json"%pre);C=load("%s_C.json"%pre)
    ay=st.mean([r["Y"] for r in A]);by=st.mean([r["Y"] for r in B])
    M=[r["M"] for r in B if r.get("M") and r["M"]>0]  # 仅B组,同口径
    Y=[r["Y"] for r in B if r.get("M") and r["M"]>0]
    d={"name":name,"ay":ay,"by":by,"r":pearson(M,Y)}
    if C:  # 双刃剑用C组
        d["cd_fact"]=dim_mean(C,"factual")-dim_mean(A,"factual")
        d["cd_depth"]=dim_mean(C,"depth")-dim_mean(A,"depth")
    D.append(d)

fig,axes=plt.subplots(1,2,figsize=(14,5.5))

# 图A: A/B组Y分组柱 + r点(次轴), 客观并列不加箭头
ax1=axes[0];x=range(len(D));w=0.35
ax1.bar([i-w/2 for i in x],[d["ay"] for d in D],w,label="A组(无反问)",color="#8ECAE6")
ax1.bar([i+w/2 for i in x],[d["by"] for d in D],w,label="B组(反问)",color="#FB8500")
for i,d in enumerate(D):
    ax1.text(i-w/2,d["ay"]+0.05,"%.2f"%d["ay"],ha="center",fontsize=9)
    ax1.text(i+w/2,d["by"]+0.05,"%.2f"%d["by"],ha="center",fontsize=9)
ax1.set_ylabel("回答质量 Y",fontsize=12);ax1.set_ylim(0,5)
ax1.set_xticks(list(x));ax1.set_xticklabels([d["name"] for d in D],fontsize=12)
ax2=ax1.twinx()
ax2.plot(x,[d["r"] for d in D],"D",color="#C1121F",ms=11,label="连续 r(M,Y) [仅B组]")
ax2.set_ylabel("连续 r(M,Y)",color="#C1121F",fontsize=12);ax2.set_ylim(0,0.6)
for i,d in enumerate(D): ax2.text(i+0.05,d["r"],"%.2f"%d["r"],va="center",color="#C1121F",fontsize=10,fontweight="bold")
ax1.set_title("(a) 三领域 A/B 组质量与 M–Y 相关（0.5B，同口径）\n三领域 M–Y 均正但不随约束单调；法律最弱源于能力边际递减",fontsize=11)
h1,lb1=ax1.get_legend_handles_labels();h2,lb2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,lb1+lb2,loc="upper right",fontsize=9)

# 图B: 双刃剑(C组), 仅诗词/菜谱有C
ax=axes[1]
DC=[d for d in D if "cd_fact" in d]
x=range(len(DC));w=0.35
ax.bar([i-w/2 for i in x],[d["cd_fact"] for d in DC],w,label="Δfactual 事实准确性",color="#4C72B0")
ax.bar([i+w/2 for i in x],[d["cd_depth"] for d in DC],w,label="Δdepth 分析深度",color="#55A868")
ax.axhline(0,color="#333",lw=0.8)
for i,d in enumerate(DC):
    ax.text(i-w/2,d["cd_fact"]+(0.03 if d["cd_fact"]>=0 else -0.1),"%+.2f"%d["cd_fact"],ha="center",fontsize=9)
    ax.text(i+w/2,d["cd_depth"]+(0.03 if d["cd_depth"]>=0 else -0.1),"%+.2f"%d["cd_depth"],ha="center",fontsize=9)
ax.set_ylabel("C组相对A组的维度变化 (Δ)",fontsize=12)
ax.set_xticks(list(x));ax.set_xticklabels([d["name"] for d in DC],fontsize=12)
ax.set_title("(b) 高强度反问(C组)的双刃剑效应\n古诗词：深度↑事实↓；反映反问对不同维度的分化影响",fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
out=RES/"paper_figs";out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/"fig_domain.png",dpi=300,bbox_inches="tight",facecolor="white")
print("saved fig_domain.png")
for d in D:
    extra=" Δfact=%+.2f Δdepth=%+.2f"%(d["cd_fact"],d["cd_depth"]) if "cd_fact" in d else " (无C组)"
    print("  %s: A=%.2f B=%.2f r=%.3f%s"%(d["name"],d["ay"],d["by"],d["r"],extra))
