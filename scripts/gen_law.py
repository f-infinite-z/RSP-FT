"""
民法典QA生成（法条锚定防幻觉+多线程并发+5题型×988条）。来源：民法典原文+DeepSeek API

用法：
  python gen_law.py --api-key sk-xxx --out data/law/qa_pairs/all.json --workers 15
  python gen_law.py --api-key sk-xxx --limit 50 --out data/law/qa_pairs/sample.json  # 试生成
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse, json, sys, io, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI

# 题型：概念解释 / 构成要件 / 情形辨析 / 法律后果 / 条文关联
QA_TYPES = {
    "concept": "概念解释——解释本条中的核心法律概念或术语的含义",
    "elements": "构成要件——本条规定的行为/权利/责任的成立需要满足哪些条件",
    "scenario": "情形辨析——设计一个具体生活情形，问该情形是否适用本条、如何适用",
    "consequence": "法律后果——违反本条或满足本条会产生什么法律后果",
    "relation": "要点归纳——本条最核心的规则要点是什么",
}
# 每条默认出3个题型（滚动选取，保证多样）
TYPE_CYCLE = ["scenario", "elements", "concept", "consequence", "relation"]

SYSTEM_QA = (
    "你是中国民法典领域的法律专家。给定民法典的一个条文原文，生成一个高质量问答对。"
    "严格要求：\n"
    "1. 答案必须【完全基于给定条文原文】，不得引入条文之外的编造内容，不得虚构其他法条编号；\n"
    "2. 若需引用法条，只引用给定的本条条号；\n"
    "3. 问题要具体、有实际意义；答案准确、简明、有条理；\n"
    "4. 同时生成一个'反问'（counter_question）——即回答者对自己答案的深化追问，"
    "紧扣本条的适用边界或关键要件；并给出该反问的参考回答（counter_answer），同样基于本条。\n"
    "只输出JSON：{\"question\":\"...\",\"answer\":\"...\",\"counter_question\":\"...\",\"counter_answer\":\"...\"}"
)


def gen_qa(client, model, rec, qa_type, timeout=40, retries=2):
    tiao = rec["tiao"]; text = rec["text"]; bian = rec.get("bian", "")
    usr = (
        "民法典{bian} {tiao}\n条文原文：{text}\n\n"
        "题型：{t}——{desc}\n"
        "请基于以上条文原文生成问答对。答案须严格依据本条，不得编造。"
    ).format(bian=bian, tiao=tiao, text=text, t=qa_type, desc=QA_TYPES[qa_type])
    for a in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.7, max_tokens=1200, timeout=timeout,
                messages=[{"role": "system", "content": SYSTEM_QA},
                          {"role": "user", "content": usr}],
                response_format={"type": "json_object"})
            data = json.loads(r.choices[0].message.content.strip())
            if data.get("question") and data.get("answer"):
                data["qa_type"] = qa_type
                data["law_tiao"] = tiao
                data["law_bian"] = bian
                data["law_source"] = text  # 保留原文供核验
                return data
        except Exception:
            if a < retries:
                time.sleep(2 ** a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law-data", default="data/law/minfadian_3bian.json")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--types-per-tiao", type=int, default=3, help="每条法条生成几个题型QA")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N条法条(0=全部,试生成用)")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--out", default="data/law/qa_pairs/all.json")
    args = ap.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    laws = json.load(open(args.law_data, encoding="utf-8"))
    if args.limit > 0:
        laws = laws[:args.limit]

    # 展开成 (法条, 题型) 任务
    tasks = []
    for i, rec in enumerate(laws):
        for k in range(args.types_per_tiao):
            tasks.append((rec, TYPE_CYCLE[(i + k) % len(TYPE_CYCLE)]))
    print("[gen_law] {n}条法条 x {t}题型 = {m}个QA任务".format(
        n=len(laws), t=args.types_per_tiao, m=len(tasks)), flush=True)

    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if out_path.exists():
        try:
            results = json.load(open(out_path, encoding="utf-8"))
            print("[gen_law] 断点续传：已有 {n} 条".format(n=len(results)))
        except Exception:
            results = []
    done = {(r["law_tiao"], r["qa_type"]) for r in results}
    todo = [(rec, t) for rec, t in tasks if (rec["tiao"], t) not in done]

    lock = threading.Lock(); cnt = [0]; fail = [0]

    def work(task):
        rec, qa_type = task
        qa = gen_qa(client, args.model, rec, qa_type)
        with lock:
            cnt[0] += 1
            if qa:
                results.append(qa)
            else:
                fail[0] += 1
            if cnt[0] % 50 == 0:
                print("  [{d}/{t}] 成功={s} 失败={f}".format(
                    d=cnt[0], t=len(todo), s=len(results), f=fail[0]), flush=True)
                json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[gen_law] 完成：{n} 条 QA，失败 {f}，输出 {o}".format(
        n=len(results), f=fail[0], o=args.out), flush=True)


if __name__ == "__main__":
    main()
