"""
3B三道坎诊断（天花板压缩+配对检验+M-Y连续相关+长度对照）。来源：配对t+Pearson
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, statistics as st, math
from collections import Counter
from pathlib import Path

RES = Path("results")
L = []
def log(s=""): L.append(str(s))

def load(fn):
    return json.load(open(RES / fn, encoding="utf-8"))

def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else float("nan")

def paired_t(a, b):
    d = [x-y for x, y in zip(a, b)]; n = len(d)
    m = st.mean(d); s = st.stdev(d)
    return m, m/(s/math.sqrt(n))

# ============ 数据 ============
raw = {g: load("recipe_3b_%s.json" % g) for g in "ABC"}
con = {g: load("recipe_3b_concise_%s.json" % g) for g in "ABC"}

log("=" * 60)
log("3B 版1 三道坎诊断")
log("=" * 60)

# ---- 坎1: 分数分布(天花板压缩检验) ----
log("\n【坎1】分数分布 —— 检验高分段天花板压缩")
for tag, D in [("原始", raw), ("控长度", con)]:
    log("  [%s]" % tag)
    for g in "ABC":
        ys = [r["Y"] for r in D[g]]
        hist = Counter(round(y) for y in ys)
        bars = " ".join("%d:%d" % (k, hist[k]) for k in sorted(hist))
        log("    %s: 均值%.2f sd%.2f 范围[%.1f,%.1f] | 分布 %s" % (
            g, st.mean(ys), st.pstdev(ys), min(ys), max(ys), bars))
log("  判读：若分数大量挤在5-6且sd很小→天花板压缩；sd正常(>1)且分布散→裁判有区分度")

# ---- 坎2: 配对检验(显著性) ----
log("\n【坎2】配对检验（|t|>1.98 → p<0.05 显著）")
for tag, D in [("原始", raw), ("控长度", con)]:
    log("  [%s]" % tag)
    for x, y in [("C","A"),("B","A"),("C","B")]:
        m, t = paired_t([r["Y"] for r in D[x]], [r["Y"] for r in D[y]])
        sig = "✓显著" if abs(t) > 1.98 else "✗不显著"
        log("    %s vs %s: Δ=%+.3f t=%.2f %s" % (x, y, m, t, sig))

# ---- 坎3: 连续 M-Y 相关(机制) ----
log("\n【坎3】连续 M-Y 相关（3B 第5数据集，验机制）")
for tag, D in [("原始", raw), ("控长度", con)]:
    M, Y = [], []
    for g in ("B", "C"):
        for r in D[g]:
            if r.get("M") and r["M"] > 0:
                M.append(r["M"]); Y.append(r["Y"])
    if len(M) > 3:
        r = pearson(M, Y)
        med = st.median(M)
        hi = st.mean([y for m,y in zip(M,Y) if m>=med]); lo = st.mean([y for m,y in zip(M,Y) if m<med])
        log("  [%s] n=%d r(M,Y)=%.3f 高M组Y=%.3f 低M组Y=%.3f 差=%+.3f" % (tag, len(M), r, hi, lo, hi-lo))

# ---- 补充: 长度对照(控长度不对称检验) ----
log("\n【补充】长度变化 —— 检验控长度是否'偏袒'B/C")
for g in "ABC":
    lr = st.mean([len(r["eval_answer"]) for r in raw[g]])
    lc = st.mean([len(r["eval_answer"]) for r in con[g]])
    yr = st.mean([r["Y"] for r in raw[g]]); yc = st.mean([r["Y"] for r in con[g]])
    log("  %s: 长度 %d→%d (%+d)  Y %.3f→%.3f (%+.3f)" % (g, lr, lc, lc-lr, yr, yc, yc-yr))
log("  判读：A短答控长度掉分(硬砍) vs B/C长答控长度去冗余；看Y变化是否与长度变化解耦")

report = "\n".join(L)
(RES / "recipe_3b_diagnosis.txt").write_text(report, encoding="utf-8")
Path("_diag.txt").write_text(report, encoding="utf-8")
print("done")
