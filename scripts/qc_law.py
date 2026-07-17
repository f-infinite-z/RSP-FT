"""
法律QA程序化质检（超短/答非所问/引用错误条号/含糊免责检测）。来源：规则匹配
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, random, re
from pathlib import Path

d = json.load(open("data/law/qa_pairs/all.json", encoding="utf-8"))
L = []
def log(s=""): L.append(str(s))

def cn_num(s):
    # 提取"第X条"里的中文数字串
    m = re.search(r'第([一二三四五六七八九十百千零两]+)条', s)
    return m.group(1) if m else None

suspect = {"超短答案": [], "答案未引本条且泛化": [], "答非所问疑似": [],
           "反问缺失或过短": [], "答案含免责/无法回答": [], "引用了其他条号": []}

for i, r in enumerate(d):
    q = r.get("question", ""); a = r.get("answer", "")
    cq = r.get("counter_question", ""); src = r.get("law_source", "")
    tiao = r.get("law_tiao", "")
    # 1 超短答案
    if len(a) < 20:
        suspect["超短答案"].append(i)
    # 2 反问缺失/过短
    if not cq or len(str(cq)) < 6:
        suspect["反问缺失或过短"].append(i)
    # 3 答案含"无法/未明确/不清楚/无法回答"等含糊词
    if re.search(r'无法回答|无法确定|未明确规定|不清楚|没有相关|无法提供', a):
        suspect["答案含免责/无法回答"].append(i)
    # 4 答案引用了"第X条"但和本条号不一致(可能张冠李戴)
    tiao_num = cn_num(tiao)
    for m in re.findall(r'第([一二三四五六七八九十百千零两]+)条', a):
        if tiao_num and m != tiao_num:
            suspect["引用了其他条号"].append(i)
            break
    # 5 答非所问疑似:答案与问题关键词几乎无重叠(粗筛)
    qkey = set(re.findall(r'[\u4e00-\u9fa5]{2,}', q))
    akey = set(re.findall(r'[\u4e00-\u9fa5]{2,}', a))
    if qkey and len(qkey & akey) == 0 and len(a) > 20:
        suspect["答非所问疑似"].append(i)

log("=" * 60)
log("法律QA 程序化质检（2964条）")
log("=" * 60)
for k, v in suspect.items():
    log("  %s: %d 条  %s" % (k, len(v), v[:15] if v else ""))

total_suspect = set()
for v in suspect.values():
    total_suspect |= set(v)
log("\n可疑条目合计（去重）: %d 条 (%.1f%%)" % (len(total_suspect), 100*len(total_suspect)/len(d)))
log("即：%.1f%% 通过程序化质检" % (100*(1-len(total_suspect)/len(d))))

# 导出可疑条目详情供人工看
log("\n" + "=" * 60)
log("【可疑条目详情】（这些是需要人工重点看的）")
for i in sorted(total_suspect)[:30]:
    r = d[i]
    log("-" * 50)
    log("#%d [%s|%s] %s" % (i, r.get("law_bian",""), r.get("law_tiao",""), r.get("qa_type","")))
    log("  问: %s" % r.get("question","")[:70])
    log("  答: %s" % r.get("answer","")[:120])
    log("  原文: %s" % r.get("law_source","")[:70])

# 随机抽15条正常的供交叉核对
random.seed(11)
normal = [i for i in range(len(d)) if i not in total_suspect]
log("\n" + "=" * 60)
log("【随机抽15条(通过质检的)供交叉核对】")
for i in random.sample(normal, 15):
    r = d[i]
    log("-" * 50)
    log("#%d [%s] %s" % (i, r.get("law_tiao",""), r.get("qa_type","")))
    log("  问: %s" % r.get("question","")[:70])
    log("  答: %s" % r.get("answer","")[:120])

Path("_law_qc.txt").write_text("\n".join(L), encoding="utf-8")
print("done, 可疑%d/%d" % (len(total_suspect), len(d)))
