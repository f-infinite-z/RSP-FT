#!/bin/bash
# 整夜串联2
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 鏁村涓茶仈2锛圱1瀹屾垚鍚庢寕锛夛細T1-plus(0.5B) + 1.5B T1-plus + T4姝ラ1
# 涓変釜閮芥槸璇勬祴/API绫伙紝鏃犺缁冮闄╋紝鍙暣澶滄棤浜鸿窇
# 鈿狅笍 鍓嶆彁锛歍1(绾帶闀垮害)宸茶窇瀹岋紝GPU绌洪棽
# 鐢ㄦ硶锛歯ohup bash run_overnight2.sh > overnight2.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

echo "########## 鏁村涓茶仈2 寮€濮?$(date) ##########"

# ===== [0/3] 琛?T1绾帶闀垮害 鐨勫姣?CoT璇勬祴锛坮ecipe_concise宸茬敱T1鍗曠粍璇勬祴鐢熸垚锛?====
echo "===== [0/3] T1绾帶闀垮害 瀵规瘮+CoT璇勬祴 $(date) ====="
if [ -f results/recipe_concise_A.json ] && [ -f results/recipe_concise_B.json ] && [ -f results/recipe_concise_C.json ]; then
  python -m src.eval.run_eval --domain recipe --compare \
    results/recipe_concise_A.json results/recipe_concise_B.json results/recipe_concise_C.json \
    --judges-config configs/judges.yaml --out results/recipe_concise_compare.json
  tar czf t1_concise_results.tar.gz results/recipe_concise_*.json
else
  echo "  [璺宠繃] recipe_concise_A/B/C 鏈叏灏辩华"
fi

# ===== [1/3] T1-plus 0.5B锛堟帶闀垮害+鎵ｉ锛孋ONCISE_STRICT锛?====
echo "===== [1/3] T1-plus 0.5B $(date) ====="
export CONCISE_STRICT=1
export CONCISE_LIMIT=150
unset CONCISE_MODE
for G in A B C; do
  echo "--- [T1plus-0.5B $G] $(date +%H:%M:%S) ---"
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_lr5_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_strict_$G.json
done
echo "--- [T1plus-0.5B] 瀵规瘮+CoT璇勬祴 $(date +%H:%M:%S) ---"
python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_strict_A.json results/recipe_strict_B.json results/recipe_strict_C.json \
  --judges-config configs/judges.yaml --out results/recipe_strict_compare.json
tar czf t1plus_0.5b_results.tar.gz results/recipe_strict_*.json

# ===== [2/3] T1-plus 1.5B锛堟帶闀垮害+鎵ｉ锛?====
echo "===== [2/3] T1-plus 1.5B $(date) ====="
for G in A B C; do
  echo "--- [T1plus-1.5B $G] $(date +%H:%M:%S) ---"
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_1.5b_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_1.5b_strict_$G.json
done
echo "--- [T1plus-1.5B] 瀵规瘮+CoT璇勬祴 $(date +%H:%M:%S) ---"
python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_1.5b_strict_A.json results/recipe_1.5b_strict_B.json results/recipe_1.5b_strict_C.json \
  --judges-config configs/judges.yaml --out results/recipe_1.5b_strict_compare.json
tar czf t1plus_1.5b_results.tar.gz results/recipe_1.5b_strict_*.json

# ===== [3/3] T4姝ラ1锛堢敓鎴?妗ｅ弽闂?鎵撳垎+鎿嶇旱妫€楠岋級=====
echo "===== [3/3] T4姝ラ1 鐢熸垚+鎵撳垎 $(date) ====="
unset CONCISE_STRICT
unset CONCISE_MODE
python t4_gen_grade.py --limit 500 --api-key "$KEY" \
  --judges-config configs/judges.yaml --workers 20 \
  --out results/t4_cq_graded.json

echo "########## 鏁村涓茶仈2 鍏ㄩ儴瀹屾垚 $(date) ##########"
echo "鐫￠啋鐪嬶細"
echo "  1. T1-plus 0.5B: recipe_strict_* (涓夋柟瀵规瘮鍘熷/鎺ч暱搴?鎺ч暱搴?鎵ｉ)"
echo "  2. T1-plus 1.5B: recipe_1.5b_strict_*"
echo "  3. T4鎿嶇旱妫€楠? t4_gen_grade.py杈撳嚭鐨?妗鍧囧€?鐪媣low<low<mid<high鏄惁鍗曡皟)"
echo "涓嬭浇: t1plus_0.5b_results.tar.gz + t1plus_1.5b_results.tar.gz + results/t4_cq_graded*.json"

