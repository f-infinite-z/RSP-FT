#!/bin/bash
# 整夜串联1
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 鏁村涓茶仈锛堢潯鍓嶆寕锛夛細T4姝ラ2-4 + T1-1.5B鎺ч暱搴?# 鈿狅笍 鍓嶆彁锛歍4姝ラ1(t4_gen_grade.py)宸插畬鎴?涓?鎿嶇旱妫€楠岄€氳繃(vlow<low<mid<high鍗曡皟)
#         鐫″墠纭鎿嶇旱鎴愬姛鍚庡啀鎸傛湰鑴氭湰
# 鐢ㄦ硶锛歯ohup bash run_overnight.sh > overnight.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

echo "########## 鏁村涓茶仈寮€濮?$(date) ##########"

# ===== 闃舵1锛歍4 姝ラ2 鐢熸垚3妗ｈ缁冮泦 =====
echo "===== [1/5] T4鐢熸垚3妗ｈ缁冮泦 $(date) ====="
python t4_make_tiers.py --graded results/t4_cq_graded_3tier.json \
  --base data/recipe/qa_pairs_clean/train.json --out-dir data/recipe/qa_pairs_clean

# ===== 闃舵2锛歍4 姝ラ2.5 涓?妗ｅ弽闂鐢熸垚M_A缂撳瓨 =====
echo "===== [2/5] T4棰勭敓鎴?妗ｇ紦瀛?$(date) ====="
for T in low mid high; do
  python pregen_counter_answers.py --domain recipe \
    --train-data data/recipe/qa_pairs_clean/t4_$T.json --limit 500 \
    --api-key "$KEY" --workers 30 --out cache/t4_${T}_cache.json
done

# ===== 闃舵3锛歍4 姝ラ3 涓夋。self_play璁粌(0.5B, lr5e-6+浣欏鸡+5杞?鎵归噺) =====
echo "===== [3/5] T4涓夋。璁粌 $(date) ====="
for T in low mid high; do
  echo "--- T4璁粌 $T 妗?$(date +%H:%M:%S) ---"
  python -m src.training.self_play --sft-model models/recipe_sft \
    --train-data data/recipe/qa_pairs_clean/t4_$T.json --domain recipe --group B \
    --iterations 5 --lr 5e-6 --lr-schedule cosine --batch-size 8 \
    --train-limit 500 --gen-batch 48 --ma-cache cache/t4_${T}_cache.json \
    --api-key "$KEY" --out models/t4_$T
done

# ===== 闃舵4锛歍4 姝ラ4 涓夋。璇勬祴 =====
echo "===== [4/5] T4涓夋。璇勬祴 $(date) ====="
for T in low mid high; do
  python -m src.eval.run_eval --domain recipe --group B \
    --model models/t4_$T/groupB_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/t4_$T.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/t4_$T.json
done
tar czf t4_results.tar.gz results/t4_*.json results/t4_cq_graded*.json

# ===== 闃舵5锛歍1-1.5B 鎺ч暱搴﹁瘎娴?=====
echo "===== [5/5] T1-1.5B鎺ч暱搴﹁瘎娴?$(date) ====="
export CONCISE_MODE=1
export CONCISE_LIMIT=150
for G in A B C; do
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_1.5b_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_1.5b_concise_$G.json
done
tar czf t1_1.5b_results.tar.gz results/recipe_1.5b_concise_*.json

echo "########## 鏁村涓茶仈鍏ㄩ儴瀹屾垚 $(date) ##########"
echo "涓嬭浇锛歵4_results.tar.gz + t1_1.5b_results.tar.gz"

