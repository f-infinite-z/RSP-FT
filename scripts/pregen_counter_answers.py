"""
M_A补充库批量预生成（sha1查表+并发缓存+断点续传+失败记录）。来源：DeepSeek API

问题：self_play B/C 组训练时，每条样本都实时串行调 DeepSeek API 生成 M_A 补充，
      GPU 干等网络，1500条×3轮 API 调用极慢且烧预算。

方案：训练前用本脚本【并发】批量生成所有 (question, counter_question) → 补充r，
      存成缓存 json；训练时 CounterAnswerer 查表命中，零 API 等待。

覆盖的反问来源（与 self_play 一致）：
  - B 组：数据集 counter_question 字段
  - C 组：黄金三问全部 3 条（因 pick 用 hash 轮转，全缓存以保证命中）

缓存 key：sha1(question + "|||" + counter_question)，与 CounterAnswerer 查表口径一致。

内置健壮性：
  - 可控并发（--workers，默认 8，保守防限流）
  - 单请求超时（--timeout，默认 30s）
  - 429/异常 指数退避重试（--retries，默认 3）
  - 断点续传（每 --save-every 条原子写盘，重启跳过已完成）
  - 失败记录（写入 <out>.failed.json，可事后补跑）

用法：
  # 小批测并发（先确认 key 能扛多少并发，不触发 429）
  python pregen_counter_answers.py --domain poetry \
      --train-data data/poetry/qa_pairs/train.json --limit 50 \
      --api-key sk-xxx --workers 8 --out cache/poetry_ma_cache.json

  # 全量预生成
  python pregen_counter_answers.py --domain poetry \
      --train-data data/poetry/qa_pairs/train.json --limit 1500 \
      --api-key sk-xxx --workers 10 --out cache/poetry_ma_cache.json
  python pregen_counter_answers.py --domain recipe \
      --train-data data/recipe/qa_pairs_clean/train.json --limit 1500 \
      --api-key sk-xxx --workers 10 --out cache/recipe_ma_cache.json
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.counter_question.golden_questions import GoldenQuestions

DOMAIN_NAME = {"recipe": "中华菜谱与烹饪", "poetry": "古诗词鉴赏"}


def cache_key(question: str, cq: str) -> str:
    return hashlib.sha1((question + "|||" + cq).encode("utf-8")).hexdigest()


def collect_pairs(data, domain):
    """收集所有需要生成补充的 (question, cq) 对（B组反问 + C组黄金三问）。"""
    golden = GoldenQuestions(domain)
    pairs = {}  # key -> (question, cq)
    for item in data:
        q = item.get("question", "")
        if not q:
            continue
        # B 组：数据集普通反问
        cq_b = str(item.get("counter_question", "")).strip()
        if cq_b:
            pairs[cache_key(q, cq_b)] = (q, cq_b)
        # C 组：黄金三问全部 3 条
        for cq_c in golden.fill(item):
            if cq_c:
                pairs[cache_key(q, cq_c)] = (q, cq_c)
    return pairs


def make_client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def gen_one(client, domain, question, cq, model, timeout, retries):
    dom = DOMAIN_NAME.get(domain, domain)
    sys_p = f"你是{dom}领域专家。请简洁、准确地回答下面的追问，2-4句话。"
    usr = f"背景问题：{question}\n追问：{cq}"
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.5, max_tokens=400, timeout=timeout,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": usr}])
            return resp.choices[0].message.content.strip(), None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            is_rate = "429" in str(e) or "rate" in str(e).lower()
            if attempt < retries:
                # 429 用更长退避
                time.sleep((3 if is_rate else 1) * (2 ** attempt))
                continue
            return None, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["poetry", "recipe", "law"])
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N条数据(0=全部)")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数(保守防限流)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    data = json.load(open(args.train_data, encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    pairs = collect_pairs(data, args.domain)
    print(f"[pregen] {args.domain}: {len(data)}条数据 → {len(pairs)}个唯一(问题,反问)对")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：加载已有缓存
    cache = {}
    if out_path.exists():
        try:
            cache = json.load(open(out_path, encoding="utf-8"))
            print(f"[pregen] 断点续传：已有 {len(cache)} 条缓存，跳过")
        except Exception:
            cache = {}

    todo = {k: v for k, v in pairs.items() if k not in cache}
    print(f"[pregen] 待生成 {len(todo)} 条（并发={args.workers}, 超时={args.timeout}s）")
    if not todo:
        print("[pregen] 全部已缓存，无需生成")
        return

    client = make_client(args.api_key)
    lock = threading.Lock()
    failed = {}
    done = [0]
    t0 = time.time()

    def worker(kv):
        key, (q, cq) = kv
        ans, err = gen_one(client, args.domain, q, cq, args.model, args.timeout, args.retries)
        with lock:
            if ans is not None:
                cache[key] = ans
            else:
                failed[key] = {"question": q, "cq": cq, "error": err}
            done[0] += 1
            n = done[0]
            if n % 10 == 0 or n == len(todo):
                rate = n / max(1e-6, time.time() - t0)
                eta = (len(todo) - n) / max(1e-6, rate)
                print(f"  [{n}/{len(todo)}] 成功={len(cache)} 失败={len(failed)} "
                      f"速率={rate:.1f}/s ETA={eta/60:.1f}min", flush=True)
            if n % args.save_every == 0:
                _atomic_save(out_path, cache)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(worker, todo.items()))

    _atomic_save(out_path, cache)
    if failed:
        fp = str(out_path) + ".failed.json"
        json.dump(failed, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[pregen] {len(failed)} 条失败，已记录到 {fp}（可重跑本命令自动补）")

    dt = time.time() - t0
    print(f"[pregen] 完成：缓存 {len(cache)} 条，用时 {dt/60:.1f}min，"
          f"平均 {len(todo)/max(1e-6,dt):.1f}条/秒")
    print(f"[pregen] 缓存已存至 {out_path}")


def _atomic_save(path: Path, cache: dict):
    tmp = str(path) + ".tmp"
    json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    Path(tmp).replace(path)


if __name__ == "__main__":
    main()
