"""
QA 数据质量抽检脚本

本脚本由 AI 辅助生成初稿，经人工适配、调试、校验。
核心算法与实验设计由作者主导完成。支持两种模式：
  1. 自动结构检查（--auto）：字段完整性、长度、重复、题型分布
  2. LLM 辅助评分抽检（--llm）：调用 DeepSeek 对随机抽样打分
  3. 人工抽检导出（--sample N）：随机抽样导出为便于人工审阅的文本

用法：
  python -m src.utils.quality_check --file data/recipe/qa_pairs/train.json --auto
  python -m src.utils.quality_check --file data/poetry/qa_pairs/train.json --sample 100 --out check_sample.txt
  python -m src.utils.quality_check --file data/recipe/qa_pairs/train.json --llm --n 50 --api-key sk-xxx
"""
import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = ["question", "answer", "qa_type", "counter_question", "counter_answer"]

MIN_Q_LEN = 5
MIN_A_LEN = 20
MIN_CQ_LEN = 5
MIN_CA_LEN = 10


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_check(data):
    """自动结构检查，返回问题报告"""
    n = len(data)
    issues = {
        "missing_field": [],
        "empty_field": [],
        "too_short_q": [],
        "too_short_a": [],
        "too_short_cq": [],
        "too_short_ca": [],
        "q_eq_cq": [],
    }
    q_seen = {}
    dup_q = []

    for i, item in enumerate(data):
        for fld in REQUIRED_FIELDS:
            if fld not in item:
                issues["missing_field"].append((i, fld))
            elif not str(item.get(fld, "")).strip():
                issues["empty_field"].append((i, fld))

        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        cq = str(item.get("counter_question", "")).strip()
        ca = str(item.get("counter_answer", "")).strip()

        if len(q) < MIN_Q_LEN:
            issues["too_short_q"].append(i)
        if len(a) < MIN_A_LEN:
            issues["too_short_a"].append(i)
        if len(cq) < MIN_CQ_LEN:
            issues["too_short_cq"].append(i)
        if len(ca) < MIN_CA_LEN:
            issues["too_short_ca"].append(i)
        if q and q == cq:
            issues["q_eq_cq"].append(i)

        if q:
            if q in q_seen:
                dup_q.append((i, q_seen[q]))
            else:
                q_seen[q] = i

    qa_types = Counter(item.get("qa_type", "?") for item in data)

    print(f"=== 自动结构检查：共 {n} 条 ===")
    total_issues = 0
    for name, lst in issues.items():
        if lst:
            total_issues += len(lst)
            print(f"  [{name}] {len(lst)} 条")
            for x in lst[:3]:
                print(f"      e.g. idx {x}")
    print(f"  [duplicate_question] {len(dup_q)} 条")
    total_issues += len(dup_q)

    print(f"\n  题型分布：")
    for t, c in qa_types.most_common():
        print(f"      {t}: {c} ({100*c/n:.1f}%)")

    # 额外维度分布（菜系/朝代）
    for extra in ("cuisine", "poem_dynasty"):
        vals = Counter(item.get(extra) for item in data if item.get(extra))
        if vals:
            print(f"\n  {extra} 分布：")
            for v, c in vals.most_common():
                print(f"      {v}: {c}")

    avg_a = sum(len(str(x.get("answer", ""))) for x in data) / n
    avg_cq = sum(len(str(x.get("counter_question", ""))) for x in data) / n
    print(f"\n  平均答案长度: {avg_a:.0f} 字 | 平均反问长度: {avg_cq:.0f} 字")

    rate = 100 * total_issues / n if n else 0
    print(f"\n  问题条目总数: {total_issues} ({rate:.1f}%)")
    print(f"  结论: {'通过 (问题率<15%)' if rate < 15 else '需重做 (问题率>=15%)'}")
    return total_issues, rate


def export_sample(data, n, out_path):
    """随机抽样导出为便于人工审阅的文本"""
    sample = random.sample(data, min(n, len(data)))
    lines = []
    for i, item in enumerate(sample, 1):
        lines.append(f"{'='*70}\n[#{i}] 题型: {item.get('qa_type','?')}"
                     + (f" | {item.get('cuisine','')}" if item.get('cuisine') else "")
                     + (f" | {item.get('poem_author','')}·{item.get('poem_title','')}" if item.get('poem_author') else ""))
        lines.append(f"Q : {item.get('question','')}")
        lines.append(f"A : {item.get('answer','')}")
        lines.append(f"CQ: {item.get('counter_question','')}")
        lines.append(f"CA: {item.get('counter_answer','')}")
        lines.append("[ ] 正确  [ ] 有误  批注: ____________________\n")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出 {len(sample)} 条抽样至 {out_path}")


def llm_check(data, n, api_key, domain):
    """调用 DeepSeek 对抽样 QA 打分（准确性/完整性），统计合格率"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    sample = random.sample(data, min(n, len(data)))
    domain_desc = {"recipe": "中华菜谱与烹饪", "poetry": "古诗词鉴赏"}.get(domain, domain)

    sys_prompt = (f"你是{domain_desc}领域专家。评估给定问答对的质量，输出JSON："
                  '{"accuracy":1-5,"completeness":1-5,"relevant":true/false,"issue":"简述问题或none"}。只输出JSON。')
    scores = []
    fails = []
    for i, item in enumerate(sample):
        user = f"问题：{item.get('question','')}\n回答：{item.get('answer','')}"
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat", temperature=0, max_tokens=300,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"})
            r = json.loads(resp.choices[0].message.content)
            r["_q"] = item.get("question", "")[:40]
            scores.append(r)
            if r.get("accuracy", 5) < 3 or not r.get("relevant", True):
                fails.append(r)
        except Exception as e:
            print(f"  [{i}] 评分失败: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  已评 {i+1}/{len(sample)}", flush=True)

    if scores:
        avg_acc = sum(s.get("accuracy", 0) for s in scores) / len(scores)
        avg_comp = sum(s.get("completeness", 0) for s in scores) / len(scores)
        fail_rate = 100 * len(fails) / len(scores)
        print(f"\n=== LLM 抽检：{len(scores)} 条 ===")
        print(f"  平均准确性: {avg_acc:.2f}/5 | 平均完整性: {avg_comp:.2f}/5")
        print(f"  不合格 (acc<3 或不相关): {len(fails)} ({fail_rate:.1f}%)")
        print(f"  结论: {'通过' if fail_rate < 15 else '需重做'}")
        for f in fails[:5]:
            print(f"    ✗ {f.get('_q')} | {f.get('issue')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="quality_sample.txt")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--domain", default="")
    args = ap.parse_args()

    data = load(args.file)
    domain = args.domain or ("recipe" if "recipe" in args.file else "poetry")

    if args.auto or (not args.llm and not args.sample):
        auto_check(data)
    if args.sample:
        export_sample(data, args.sample, args.out)
    if args.llm:
        if not args.api_key:
            print("需要 --api-key 才能进行 LLM 抽检")
        else:
            llm_check(data, args.n, args.api_key, domain)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
