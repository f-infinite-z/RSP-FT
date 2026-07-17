"""
M-Y跨规模边际递减诊断（H1-H5：Y方差/M方差/长度混淆/裁判视角）。来源：OLS+方差分析
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, statistics as st, math
from pathlib import Path
RES = Path("results")

def load(fn):
    p = RES / fn
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

def pearson(xs, ys):
    if len(xs) < 3: return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx and sy else float("nan")

SCALES = {
    "0.5B": ("recipe_A.json", "recipe_B.json", "recipe_C.json"),
    "1.5B": ("recipe_1.5b_A.json", "recipe_1.5b_B.json", "recipe_1.5b_C.json"),
    "3B":   ("recipe_3b_A.json", "recipe_3b_B.json", "recipe_3b_C.json"),
}
L = []
def log(s=""): L.append(str(s))

log("="*66)
log("3B 上 M-Y 减弱的跨规模拆解诊断（H1-H5）")
log("="*66)

# 收集每个规模的 A/B/C 数据
data = {}
for sc, files in SCALES.items():
    d = {g: load(f) for g, f in zip("ABC", files)}
    if any(v is None for v in d.values()):
        log("%s: 缺文件，跳过" % sc); continue
    data[sc] = d

log("\n【H1】Y 方差（天花板压缩？3B若明显更小→支持）+ 【H3】A组绝对分（3B若已很高→能力主导）")
log("规模 | A组Y(方差) | B组Y(方差) | C组Y(方差) | B+C的Y方差")
for sc, d in data.items():
    line = "%-5s" % sc
    bc_y = []
    for g in "ABC":
        ys = [r["Y"] for r in d[g]]
        line += " | %.2f(%.2f)" % (st.mean(ys), st.pvariance(ys))
        if g in "BC": bc_y += ys
    line += " | %.3f" % st.pvariance(bc_y)
    log(line)

log("\n【H2】M 方差（自变量变异？3B若M挤高位方差小→测不出关系）")
log("规模 | B组M(方差) | C组M(方差) | B+C的M方差 | M范围")
for sc, d in data.items():
    bc_m = []
    parts = "%-5s" % sc
    for g in "BC":
        ms = [r["M"] for r in d[g] if r.get("M") and r["M"] > 0]
        parts += " | %.2f(%.2f)" % (st.mean(ms), st.pvariance(ms))
        bc_m += ms
    parts += " | %.3f | [%.1f,%.1f]" % (st.pvariance(bc_m), min(bc_m), max(bc_m))
    log(parts)

log("\n【核心】连续 M-Y 相关 跨规模（B+C 合并）")
log("规模 | n | r(M,Y) | 高M组Y | 低M组Y | 差")
for sc, d in data.items():
    M, Y = [], []
    for g in "BC":
        for r in d[g]:
            if r.get("M") and r["M"] > 0:
                M.append(r["M"]); Y.append(r["Y"])
    r = pearson(M, Y); med = st.median(M)
    hi = st.mean([y for m,y in zip(M,Y) if m>=med]); lo = st.mean([y for m,y in zip(M,Y) if m<med])
    log("%-5s | %d | %.3f | %.3f | %.3f | %+.3f" % (sc, len(M), r, hi, lo, hi-lo))

log("\n【H4】3B 长度混淆：长度-Y 相关 + M-长度 相关（B+C）")
if "3B" in data:
    d = data["3B"]; Ln, Y, M = [], [], []
    for g in "BC":
        for r in d[g]:
            if r.get("M") and r["M"] > 0:
                Ln.append(len(r["eval_answer"])); Y.append(r["Y"]); M.append(r["M"])
    log("  r(长度,Y)=%.3f  r(M,长度)=%.3f  r(M,Y)=%.3f" % (
        pearson(Ln, Y), pearson(M, Ln), pearson(M, Y)))
    log("  解读：若r(M,长度)强负 & r(长度,Y)强负 → M高的答案偏短/长被长度效应抵消")

log("\n【H5】3B 三裁判各自的 M-Y 相关（某裁判是否拖累）")
if "3B" in data:
    d = data["3B"]
    for j in ["deepseek", "doubao", "qwen"]:
        M, Y = [], []
        for g in "BC":
            for r in d[g]:
                pm = r.get("M_detail", {}).get("per_judge", {}).get(j)
                py = r.get("Y_detail", {}).get("per_judge", {}).get(j)
                if pm and py and pm.get("overall") and py.get("overall"):
                    M.append(pm["overall"]); Y.append(py["overall"])
        log("  %s: r(M,Y)=%.3f (n=%d)" % (j, pearson(M, Y), len(M)))

report = "\n".join(L)
(RES / "recipe_my_scale_diag.txt").write_text(report, encoding="utf-8")
Path("_myscale.txt").write_text(report, encoding="utf-8")
print("done")
