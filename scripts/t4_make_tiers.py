"""
T4步骤2（3档分层数据→替换counter_question字段→3份训练集）。来源：JSON数据处理

每档一份训练集：500道菜相同，只是 counter_question 字段替换为该菜在该档的反问。
这样用 self_play 的 B组模式(用数据集counter_question)即可训练，无需改训练代码。

产出：
  data/recipe/qa_pairs_clean/t4_low.json / t4_mid.json / t4_high.json
每份含原菜的 question/answer/dish_name + 该档 counter_question(+M值记录)。

用法：python t4_make_tiers.py --graded results/t4_cq_graded_3tier.json \
    --base data/recipe/qa_pairs_clean/train.json --out-dir data/recipe/qa_pairs_clean
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", default="results/t4_cq_graded_3tier.json")
    ap.add_argument("--base", default="data/recipe/qa_pairs_clean/train.json")
    ap.add_argument("--out-dir", default="data/recipe/qa_pairs_clean")
    args = ap.parse_args()

    graded = json.load(open(args.graded, encoding="utf-8"))
    base = json.load(open(args.base, encoding="utf-8"))
    base_idx = {r["question"]: r for r in base}

    out = Path(args.out_dir)
    for tier in ["low", "mid", "high"]:
        recs = []
        for g in graded:
            src = base_idx.get(g["question"])
            if not src:
                continue
            r = dict(src)  # 复制原菜的 question/answer/dish_name 等
            r["counter_question"] = g[tier]["cq"]   # 替换为该档反问
            r["_tier"] = tier
            r["_cq_M"] = g[tier]["M"]
            r["_cq_src"] = g[tier]["src"]
            recs.append(r)
        fp = out / f"t4_{tier}.json"
        json.dump(recs, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        import statistics as st
        ms = [g[tier]["M"] for g in graded]
        print(f"[T4] {tier}档: {len(recs)}条 → {fp}  (M均值={st.mean(ms):.3f})")

    print("\n[T4] 3档训练集就绪。步骤3：用 self_play B组模式分别训练3档：")
    print("  for T in low mid high; do")
    print("    python -m src.training.self_play --sft-model models/recipe_sft \\")
    print("      --train-data data/recipe/qa_pairs_clean/t4_$T.json --domain recipe --group B \\")
    print("      --iterations 5 --lr 5e-6 --lr-schedule cosine --batch-size 8 \\")
    print("      --train-limit 500 --gen-batch 48 --ma-cache cache/recipe_ma_cache.json \\")
    print("      --api-key $KEY --out models/t4_$T")
    print("  done")


if __name__ == "__main__":
    main()
