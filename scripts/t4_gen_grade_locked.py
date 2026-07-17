"""
T4锁文体版（4档统一大模型生成+锁文体/长度/句式）。来源：DeepSeek API
改法：4档全部由大模型生成，prompt指定质量档位+统一约束(单句/20-40字/句式对齐)。
生成后三裁判打M分 → 操纵检验(确认vlow<low<mid<high) + 字数分布自检。

用法：python t4_gen_grade_locked.py --limit 500 --api-key sk-xxx --workers 20
产出：results/t4_cq_graded_locked.json（供 t4_tertile_split.py 分层用）
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse, json, random, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openai import OpenAI
from src.judge.judge_factory import build_judges

_LOCK = (
    "要求：单句疑问句，字数控制在20-40字（超出无效）。\n"
    "句式：'针对{具体食材/工艺}，{追问}？'\n"
    "不要加任何前缀（如'问：'、'追问：'），直接输出追问文本。"
)

PROMPTS = {
    "vlow": (
        "你是烹饪初级爱好者。针对下面这道菜和它的问答，提出一个泛泛而谈的追问。"
        "追问应表面、大众化、不需要专业知识就可提出。不要涉及具体食材名称或工艺细节。\n"
    ) + _LOCK,
    "low": (
        "你是烹饪学习者。针对下面这道菜和它的问答，提出一个有一定方向但不够深入的追问。"
        "追问应提到具体食材或工艺，但不追问因果或操作细节。\n"
    ) + _LOCK,
    "mid": (
        "你是烹饪业余爱好者。针对下面这道菜和它的问答，提出一个中等质量的追问。"
        "追问应针对具体食材或工艺，有一定针对性但不过分深入。\n"
    ) + _LOCK,
    "high": (
        "你是烹饪专家。针对下面这道菜和它的问答，提出一个高质量的深度追问。"
        "追问应针对该菜的具体食材或关键工艺，追问可操作、可验证的因果关系"
        "（如某步骤如何影响口感、火候如何控制、食材替换的后果）。\n"
    ) + _LOCK,
}

def gen_cq(client, model, tier, dish, question, answer, timeout=30, retries=2):
    msg = "菜名：{dish}\n问题：{q}\n回答：{ans}".format(
        dish=dish, q=question, ans=answer[:300])
    for a in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.7, max_tokens=256, timeout=timeout,
                messages=[{"role": "system", "content": PROMPTS[tier]},
                          {"role": "user", "content": msg}])
            text = r.choices[0].message.content.strip()
            # 去除可能的前缀，取首句问句；不做字数硬过滤（靠prompt引导+事后自检）
            text = text.split("\n")[0].strip()
            for pre in ("追问：", "追问:", "问：", "问:", "反问：", "反问:"):
                if text.startswith(pre):
                    text = text[len(pre):].strip()
            if len(text) >= 6:   # 仅排除空/极短无效输出
                return text
        except Exception:
            if a < retries:
                time.sleep(2 ** a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/recipe/qa_pairs_clean/train.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--gen-model", default="deepseek-v4-flash")
    ap.add_argument("--judges-config", default="configs/judges.yaml")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="results/t4_cq_graded_locked.json")
    args = ap.parse_args()

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    data = json.load(open(args.train_data, encoding="utf-8"))[:args.limit]
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    judges = build_judges("recipe", cfg_path=args.judges_config)
    print("[T4-locked] {n}道菜 x 4档反问，裁判数={m}".format(n=len(data), m=len(judges)), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []; done_q = set()
    if out_path.exists():
        try:
            records = json.load(open(out_path, encoding="utf-8"))
            done_q = {r["question"] for r in records}
            print("[T4-locked] 断点续传：已有 {n} 条".format(n=len(records)))
        except Exception:
            records = []

    todo = [it for it in data if it.get("question") not in done_q]
    lock = threading.Lock(); cnt = [0]

    def work(item):
        dish = item.get("dish_name", "这道菜")
        q = item.get("question", ""); ans = item.get("answer", "")
        cqs = {}
        for tier in ["vlow","low","mid","high"]:
            cqs[tier] = gen_cq(client, args.gen_model, tier, dish, q, ans)

        rec = {"question": q, "dish_name": item.get("dish_name", ""), "cqs": cqs, "M": {}}
        for tier, cq in cqs.items():
            if cq is None: continue
            ms = []
            for j in judges:
                try:
                    sc = j.score_cq(q, cq)
                    if sc: ms.append(sc.overall)
                except Exception:
                    pass
            rec["M"][tier] = round(sum(ms)/len(ms), 3) if ms else None
        with lock:
            records.append(rec); cnt[0] += 1
            if cnt[0] % 20 == 0:
                print("  [{done}/{total}]".format(done=cnt[0], total=len(todo)), flush=True)
                json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 操纵检验
    import statistics as st
    print("\n===== 锁文体版 4档 M均值 =====")
    tier_ms = {t: [r["M"][t] for r in records if r["M"].get(t) is not None] for t in ["vlow","low","mid","high"]}
    for t in ["vlow","low","mid","high"]:
        vals = tier_ms[t]
        if vals:
            print("  {t}: M均值={mean:.3f} (n={n})".format(t=t, mean=st.mean(vals), n=len(vals)))

    # 字数分布自检
    print("\n===== 锁文体字数分布自检 =====")
    for t in ["vlow","low","mid","high"]:
        lens = [len(r["cqs"][t]) for r in records if r["cqs"].get(t)]
        if lens:
            print("  {t}: 字数均值={mean:.1f} 范围[{lo},{hi}]".format(
                t=t, mean=st.mean(lens), lo=min(lens), hi=max(lens)))

    print("\n[T4-locked] 产出 → {out}".format(out=args.out))
    print("[T4-locked] 下一步：run_3b_v2.sh（训练+评测）")


if __name__ == "__main__":
    main()
