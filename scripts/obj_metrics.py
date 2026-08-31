"""
客观指标评测（审稿意见5）：ROUGE-L / BERTScore / 关键词召回率 vs 裁判评分对照
作者：臧恒杰（2026-08-29）
用法：
  python -X utf8 obj_metrics.py --groups recipe_lr5 recipe_1.5b poetry law --skip-bert
  python -X utf8 obj_metrics.py --groups recipe_lr5 recipe_1.5b poetry law
输出：results/obj_metrics/ 下 每组 CSV + 汇总报告 obj_metrics_report.txt
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import jieba
from rouge_score import rouge_scorer
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "obj_metrics"

# 组配置：eval 文件 glob + 领域
GROUPS = {
    "recipe_lr5":  {"pattern": "recipe_lr5_{G}.json", "domain": "recipe"},
    "recipe_1.5b": {"pattern": "recipe_1.5b_{G}.json", "domain": "recipe"},
    "poetry":      {"pattern": "poetry_{G}.json", "domain": "poetry"},
    "law":         {"pattern": "law_{G}.json", "domain": "law"},
    "rqls":        {"pattern": "rqls_{G}.json", "domain": "recipe", "groups": ["A", "B", "C"]},
    "cqls_plus":   {"pattern": "cqls_plus_{G}.json", "domain": "recipe", "groups": ["B", "C"]},
    "cqls_strong": {"pattern": "cqls_strong_{G}.json", "domain": "recipe", "groups": ["B", "C"]},
    "pcgrad":      {"pattern": "pcgrad_{G}.json", "domain": "recipe", "groups": ["B", "C"]},
}

GOLD_FILES = {
    "recipe": ROOT / "data" / "recipe" / "qa_pairs_clean" / "test.json",
    "poetry": ROOT / "data" / "poetry" / "qa_pairs" / "train.json",
    "law":    ROOT / "data" / "law" / "qa_pairs_clean" / "test.json",
}

STOPWORDS = set("的了是在和与就不都一个上也很到说把被让对于以及或等这那之其而".strip())


class ZhTokenizer:
    """rouge_score 的中文分词器（jieba 分词后空格连接）。"""

    def tokenize(self, text: str) -> str:
        return " ".join(jieba.cut(text))


def load_gold(domain: str):
    data = json.load(open(GOLD_FILES[domain], encoding="utf-8"))
    return {x["question"]: x for x in data}


def load_eval(path: Path):
    return json.load(open(path, encoding="utf-8"))


def keyword_recall(gold_answer: str, cand: str) -> float:
    """菜谱关键词召回率：gold 中的名词性关键词在生成答案中的命中比例。"""
    import jieba.posseg as pseg
    words = set()
    for w, flag in pseg.cut(gold_answer):
        if flag.startswith("n") and len(w) > 1 and w not in STOPWORDS:
            words.add(w)
        elif flag == "x" and re.fullmatch(r"[\u4e00-\u9fa5]{2,}", w):
            words.add(w)
    words = [w for w in words if w not in STOPWORDS][:15]
    if not words:
        return 0.0
    hit = sum(1 for w in words if w in cand)
    return hit / len(words)


def compute_group(name: str, group: str, skip_bert: bool):
    cfg = GROUPS[name]
    dom = cfg["domain"]
    ev_path = ROOT / "results" / cfg["pattern"].format(G=group)
    if not ev_path.exists():
        print(f"[skip] {ev_path}")
        return None
    evs = load_eval(ev_path)
    gold = load_gold(dom)

    scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=ZhTokenizer())
    bert = None
    if not skip_bert:
        from bert_score import BERTScorer
        bert = BERTScorer(lang="zh", model_type="bert-base-chinese",
                          device="cpu", idf=False)

    rows = []
    cands, refs = [], []
    for r in evs:
        q, ans = r["question"], r.get("eval_answer") or ""
        g = gold.get(q)
        ref = g["answer"] if g else ""
        if not ref:
            continue
        rl = scorer.score(ref, ans)["rougeL"].fmeasure
        rows.append({
            "question": q, "group": group,
            "rougeL": round(rl, 4),
            "kw_recall": round(keyword_recall(ref, ans), 4),
            "Y": r.get("Y"), "M": r.get("M"),
        })
        cands.append(ans)
        refs.append(ref)

    if bert:
        P, R, F = bert.score(cands, refs)
        for row, f in zip(rows, F):
            row["bertF1"] = round(float(f), 4)

    path = OUT / f"{name}_{group}.csv"
    cols = ["question", "group", "rougeL", "kw_recall"] + \
           (["bertF1"] if not skip_bert else []) + ["Y", "M"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {name} {group}: n={len(rows)} -> {path.name}")
    return rows


def report(groups: list, skip_bert: bool):
    """汇总：组间差方向一致性 + Y/客观指标相关 + M-Y 相关"""
    lines = []
    add = lines.append
    add("=" * 72)
    add("客观指标 vs 裁判评分对照（审稿意见5）  " +
        ("无BERTScore" if skip_bert else "含BERTScore"))
    add("=" * 72)

    metrics = ["rougeL", "kw_recall"] + ([] if skip_bert else ["bertF1"])

    # 1) 各组均值表
    add("\n【一】各组客观指标与裁判 Y 均值")
    add(f"{'组':<14} {'n':>4} {'ROUGE-L':>8} {'KW召回':>7} "
        + (f"{'BERT-F1':>9} " if not skip_bert else "") + f"{'裁判Y':>7} {'M':>7}")
    stats = {}
    for name in groups:
        for g in ["A", "B", "C"]:
            rows = stats.get((name, g))
            csv_p = OUT / f"{name}_{g}.csv"
            if not csv_p.exists():
                continue
            rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
            m = {k: _mean(rows, k) for k in metrics}
            m["Y"] = _mean(rows, "Y", num=True)
            m["M"] = _mean(rows, "M", num=True)
            stats[(name, g)] = (len(rows), m)
            tag = ""
            if g == "A":
                tag = " <- 无反问"
            add(f"{name+'/'+g:<14} {len(rows):>4} {m['rougeL']:>8.4f} {m['kw_recall']:>7.3f} "
                + (f"{m['bertF1']:>9.4f} " if not skip_bert else "")
                + f"{m['Y']:>7.3f} {m['M']:>7.3f}{tag}")

    # 2) 组间差方向一致性
    add("\n【二】组间差方向一致性（Δ(C-B)、Δ(B-A) 在裁判Y vs 客观指标）")
    for name in groups:
        if (name, "A") not in stats or (name, "B") not in stats:
            continue
        add(f"--- {name} ---")
        for pair, (g1, g2) in {"C-B": ("B", "C"), "B-A": ("A", "B")}.items():
            if (name, g2) not in stats:
                add(f"  {pair}: 缺 {g2}")
                continue
            n1, m1 = stats[(name, g1)]
            n2, m2 = stats[(name, g2)]
            dy = m2["Y"] - m1["Y"]
            sigs = []
            for k in metrics:
                d = m2[k] - m1[k]
                same = "同" if (d * dy) > 0 else ("平" if abs(d) < 1e-4 else "反")
                sigs.append(f"{k}:{'%+.4f' % d}({same})")
            add(f"  {pair}: ΔY={'%+.3f' % dy} | " + " ".join(sigs))

    # 3) 相关：客观指标 vs Y（组内跨样本 Spearman）+ M vs 客观指标
    add("\n【三】Spearman 相关（跨样本）")
    add(f"{'组':<14} {'rougeL~Y':>10} {'kw~Y':>9} "
        + (f"{'bert~Y':>9} " if not skip_bert else "")
        + f"{'M~rougeL':>10} {'M~kw':>8} " + (f"{'M~bert':>9}" if not skip_bert else ""))
    for name in groups:
        for g in ["A", "B", "C"]:
            csv_p = OUT / f"{name}_{g}.csv"
            if not csv_p.exists():
                continue
            rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
            tag = ""
            if g == "A":
                tag = " <- 无反问"
            ys, ms = [], []
            r_y = {}
            for k in metrics:
                xs, yv = [], []
                for r in rows:
                    if r[k] == "" or r["Y"] == "":
                        continue
                    xs.append(float(r[k]))
                    yv.append(float(r["Y"]))
                r_y[k] = spearmanr(xs, yv).statistic if len(xs) > 5 else float("nan")
                ys.append(xs)
                ms.append([float(r["M"]) if r["M"] else 0.0 for r in rows])
            r_m = {}
            for k, kk in zip(metrics, ["M"] * len(metrics)):
                xs, mv = [], []
                for r in rows:
                    if r[k] == "" or r["M"] == "":
                        continue
                    xs.append(float(r[k]))
                    mv.append(float(r["M"]))
                r_m[k] = spearmanr(xs, mv).statistic if len(xs) > 5 else float("nan")
            add(f"{name+'/'+g:<14} {r_y['rougeL']:>10.3f} {r_y['kw_recall']:>9.3f} "
                + (f"{r_y['bertF1']:>9.3f} " if not skip_bert else "")
                + f"{r_m['rougeL']:>10.3f} {r_m['kw_recall']:>8.3f} "
                + (f"{r_m['bertF1']:>9.3f}" if not skip_bert else "") + tag)

    # 4) M-Y 相关在客观口径的对照提示
    add("\n【四】判读提示")
    add(" - 客观指标组间差与 ΔY 方向一致(同) → 裁判评分与客观质量双轨印证")
    add(" - M(裁判反问分) 与客观指标相关为正 → 反问质量高确实带来客观质量提升")
    txt = "\n".join(lines)
    report_path = OUT / "obj_metrics_report.txt"
    report_path.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n报告已保存: {report_path}")


def _mean(rows, key, num=False):
    vals = [r[key] for r in rows if r.get(key, "") != ""]
    if not vals:
        return float("nan")
    if num:
        return sum(float(v) for v in vals) / len(vals)
    return sum(float(v) for v in vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["recipe_lr5", "recipe_1.5b", "poetry", "law"])
    ap.add_argument("--skip-bert", action="store_true", help="跳过 BERTScore（提速）")
    ap.add_argument("--report-only", action="store_true", help="仅从已有 CSV 生成汇总报告")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for name in args.groups:
            grps = GROUPS[name].get("groups", ["A", "B", "C"])
            for g in grps:
                compute_group(name, g, args.skip_bert)
    report(args.groups, args.skip_bert)


if __name__ == "__main__":
    main()
