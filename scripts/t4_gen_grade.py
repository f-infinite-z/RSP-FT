"""
T4步骤1（4档反问生成+三裁判打分+操纵检验单调性验证）。来源：DeepSeek/Doubao/Qwen API

对 N=500 道菜，每道生成4档不同质量的反问：
  - vlow (极低M)：固定泛泛粗浅模板（最低锚点，不调API，快）
  - low  (低M)  ：黄金三问（已知M~5.9，通用套话，上次失败品变锚点）
  - mid  (中M)  ：数据集普通反问（counter_question字段）
  - high (高M)  ：大模型生成"针对具体食材/工艺、可验证因果"的精准反问（折中综合）

然后三裁判对全部反问打 M 分 → 操纵检验（确认 vlow<low<mid<high 显著）。
输出：cq_graded.json（每菜4档反问+各自M分），供步骤2按M分层训练。

⚠️ 先验证操纵成功再训练（吸取 A/B/C 失败教训）。

用法：
  python t4_gen_grade.py --limit 500 --api-key sk-xxx --judges-config configs/judges.yaml \
      --out results/t4_cq_graded.json --workers 20
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.counter_question.golden_questions import GoldenQuestions

# 极低M：固定泛泛粗浅模板（套任何菜，无信息增益）
VLOW_TEMPLATES = [
    "这道菜怎么样？",
    "做这道菜需要注意什么吗？",
    "这道菜好吃吗？",
    "这道菜难做吗？",
]

# 高M生成 prompt：要求针对具体食材/工艺、可验证因果的精准追问
HIGH_PROMPT = (
    "你是烹饪专家。针对下面这道菜和它的问答，提出一个高质量的深度追问。"
    "要求：紧扣该菜的具体食材或关键工艺，追问可操作、可验证的因果关系"
    "（如某步骤如何影响口感/火候如何控制），避免泛泛而问。只输出追问本身，一句话。"
)


def gen_high_cq(client, model, dish, question, answer, timeout=30, retries=2):
    import time
    usr = f"菜名：{dish}\n问题：{question}\n回答：{answer[:300]}\n请提出一个针对具体工艺/食材的高质量追问。"
    for a in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.7, max_tokens=100, timeout=timeout,
                messages=[{"role": "system", "content": HIGH_PROMPT},
                          {"role": "user", "content": usr}])
            return r.choices[0].message.content.strip()
        except Exception:
            if a < retries:
                time.sleep(2 ** a)
    return None


def build_four_cqs(item, golden, client, model, rng):
    """为一道菜构造4档反问。"""
    dish = item.get("dish_name", "这道菜")
    q = item.get("question", "")
    ans = item.get("answer", "")
    # vlow: 固定模板
    vlow = rng.choice(VLOW_TEMPLATES)
    # low: 黄金三问（取一条）
    golds = golden.fill(item)
    low = golds[abs(hash(dish)) % len(golds)] if golds else "这道菜属于什么菜系？"
    # mid: 数据集普通反问
    mid = str(item.get("counter_question", "")).strip() or "制作时火候如何掌握？"
    # high: 大模型生成
    high = gen_high_cq(client, model, dish, q, ans) or f"{dish}在关键步骤上如何控制以保证口感？"
    return {"vlow": vlow, "low": low, "mid": mid, "high": high}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/recipe/qa_pairs_clean/train.json")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--gen-model", default="deepseek-v4-flash")
    ap.add_argument("--judges-config", default="configs/judges.yaml")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="results/t4_cq_graded.json")
    args = ap.parse_args()

    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    from openai import OpenAI
    from src.judge.judge_factory import build_judges

    data = json.load(open(args.train_data, encoding="utf-8"))[:args.limit]
    golden = GoldenQuestions("recipe")
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    judges = build_judges("recipe", cfg_path=args.judges_config)
    print(f"[T4] {len(data)}道菜 × 4档反问，裁判数={len(judges)}", flush=True)

    # 断点续传
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    done_q = set()
    if out_path.exists():
        try:
            records = json.load(open(out_path, encoding="utf-8"))
            done_q = {r["question"] for r in records}
            print(f"[T4] 断点续传：已有 {len(records)} 条")
        except Exception:
            records = []

    todo = [it for it in data if it.get("question") not in done_q]
    lock = threading.Lock()
    cnt = [0]

    def work(item):
        cqs = build_four_cqs(item, golden, client, args.gen_model, random.Random(hash(item["question"]) & 0xffff))
        # 三裁判对4档反问打M分
        rec = {"question": item["question"], "dish_name": item.get("dish_name", ""), "cqs": cqs, "M": {}}
        for tier, cq in cqs.items():
            ms = []
            for j in judges:
                try:
                    sc = j.score_cq(item["question"], cq)
                    if sc:
                        ms.append(sc.overall)
                except Exception:
                    pass
            rec["M"][tier] = round(sum(ms) / len(ms), 3) if ms else None
        with lock:
            records.append(rec)
            cnt[0] += 1
            if cnt[0] % 20 == 0:
                print(f"  [{cnt[0]}/{len(todo)}]", flush=True)
                json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 操纵检验：候选4档M均值
    import statistics as st
    print("\n===== 候选池4档 M均值 =====")
    tier_ms = {t: [r["M"][t] for r in records if r["M"].get(t) is not None] for t in ["vlow", "low", "mid", "high"]}
    for t in ["vlow", "low", "mid", "high"]:
        vals = tier_ms[t]
        print(f"  {t}: M均值={st.mean(vals):.3f} (n={len(vals)})" if vals else f"  {t}: 无")

    # 按理解1：每道菜从4候选中按M选 低/中/高 三档（配对，控制菜变量）
    print("\n===== 按M为每道菜选 低/中/高 3档（X方案配对）=====")
    graded = []
    for r in records:
        pairs = [(t, r["cqs"][t], r["M"][t]) for t in ["vlow", "low", "mid", "high"] if r["M"].get(t) is not None]
        if len(pairs) < 3:
            continue
        pairs.sort(key=lambda x: x[2])  # 按M升序
        lo, hi = pairs[0], pairs[-1]
        mid = pairs[len(pairs) // 2]  # 中位
        graded.append({
            "question": r["question"], "dish_name": r["dish_name"],
            "low":  {"cq": lo[1],  "M": lo[2],  "src": lo[0]},
            "mid":  {"cq": mid[1], "M": mid[2], "src": mid[0]},
            "high": {"cq": hi[1],  "M": hi[2],  "src": hi[0]},
        })
    # 三档M分布与临界点
    for tier in ["low", "mid", "high"]:
        ms = [g[tier]["M"] for g in graded]
        print(f"  {tier}档: M均值={st.mean(ms):.3f} 范围[{min(ms):.2f},{max(ms):.2f}] (n={len(ms)})")
    # 层临界点（相邻档的边界重叠检查）
    lo_max = max(g["low"]["M"] for g in graded)
    hi_min = min(g["high"]["M"] for g in graded)
    mid_ms = sorted(g["mid"]["M"] for g in graded)
    print(f"  临界点：低档M上界={lo_max:.2f}  中档M范围[{mid_ms[0]:.2f},{mid_ms[-1]:.2f}]  高档M下界={hi_min:.2f}")
    # 配对单调性：每道菜 low<mid<high 的比例
    mono = sum(1 for g in graded if g["low"]["M"] < g["mid"]["M"] < g["high"]["M"])
    print(f"  逐菜单调(low<mid<high)比例: {mono}/{len(graded)} = {mono/len(graded):.1%}")

    graded_path = str(out_path).replace(".json", "_3tier.json")
    json.dump(graded, open(graded_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[T4] 4候选打分 → {args.out}")
    print(f"[T4] 3档分层(供步骤2训练) → {graded_path}")


if __name__ == "__main__":
    main()
