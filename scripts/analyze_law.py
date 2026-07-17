"""
法律领域Pearson/Spearman/配对检验/M-Y连续相关+M分布。来源：标准统计方法

开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, statistics as st, math
from pathlib import Path
RES = Path("results")
def load(fn): return json.load(open(RES/fn, encoding="utf-8"))
def pearson(xs, ys):
    if len(xs)<3: return float("nan")
    mx,my=st.mean(xs),st.mean(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx=math.sqrt(sum((x-mx)**2 for x in xs));sy=math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else float("nan")
def spearman(xs,ys):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]);r=[0.0]*len(v);i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[o[j+1]]==v[o[i]]:j+=1
            for t in range(i,j+1):r[o[t]]=(i+j)/2.0
            i=j+1
        return r
    return pearson(rk(xs),rk(ys))
def paired_t(a,b):
    d=[x-y for x,y in zip(a,b)];n=len(d);m=st.mean(d);s=st.stdev(d)
    return m,m/(s/math.sqrt(n))

A=load("law_A.json");B=load("law_B.json")
cA=load("law_concise_A.json");cB=load("law_concise_B.json")
L=[]
def log(s=""):L.append(str(s))

log("="*60);log("法律领域完整分析");log("="*60)

log("\n【1】A/B 均值+长度（原始 vs 控长度）")
for tag,(a,b) in [("原始",(A,B)),("控长度",(cA,cB))]:
    ya=[r['Y'] for r in a];yb=[r['Y'] for r in b]
    la=st.mean([len(r['eval_answer']) for r in a]);lb=st.mean([len(r['eval_answer']) for r in b])
    log("  [%s] A: Y=%.3f 长度%d | B: Y=%.3f 长度%d"%(tag,st.mean(ya),la,st.mean(yb),lb))

log("\n【2】A/B 配对检验（B vs A，|t|>1.98显著）")
for tag,(a,b) in [("原始",(A,B)),("控长度",(cA,cB))]:
    m,t=paired_t([r['Y'] for r in b],[r['Y'] for r in a])
    log("  [%s] B-A: Δ=%+.3f t=%.2f %s"%(tag,m,t,"✓显著" if abs(t)>1.98 else "✗不显著"))

log("\n【3】连续 M-Y 相关（B组100条，三领域普适关键）")
for tag,b in [("原始",B),("控长度",cB)]:
    M=[r['M'] for r in b if r.get('M') and r['M']>0]
    Y=[r['Y'] for r in b if r.get('M') and r['M']>0]
    if len(M)>3:
        med=st.median(M)
        hi=st.mean([y for m,y in zip(M,Y) if m>=med]);lo=st.mean([y for m,y in zip(M,Y) if m<med])
        log("  [%s] n=%d Pearson r=%.3f Spearman=%.3f 高M组Y=%.3f 低M组Y=%.3f 差=%+.3f"%(
            tag,len(M),pearson(M,Y),spearman(M,Y),hi,lo,hi-lo))

log("\n【4】长度-Y 相关（验法律长度效应方向）")
for tag,b in [("原始B",B),("原始A",A)]:
    Ln=[len(r['eval_answer']) for r in b];Y=[r['Y'] for r in b]
    log("  [%s] r(长度,Y)=%.3f"%(tag,pearson(Ln,Y)))

log("\n【5】M值分布(B组,看反问质量变异)")
M=[r['M'] for r in B if r.get('M') and r['M']>0]
log("  M均值=%.2f 方差=%.2f 范围[%.1f,%.1f]"%(st.mean(M),st.pvariance(M),min(M),max(M)))

Path("_law_analysis.txt").write_text("\n".join(L),encoding="utf-8")
print("done")
