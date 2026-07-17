#!/bin/bash
# T1控长度(1.5B)
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# T1-1.5B 鎺ч暱搴︾増锛氱敤1.5B妯″瀷(recipe_1.5b_group{A,B,C}/group{X}_iter5)寮€CONCISE_MODE閲嶈瘎娴?# 1.5B鐨凚缁勭瓟妗堣啫鑳€鏈€鏋佺(423瀛?锛屾帶闀垮害鏁堟灉鏈€鑳戒綋鐜?# 瀵规瘮锛歳ecipe_1.5b_concise_*(绮剧畝) vs recipe_1.5b_*(鍘熷)
# 鐢ㄦ硶锛歯ohup bash run_t1_1.5b.sh > t1_1.5b.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
export CONCISE_MODE=1
export CONCISE_LIMIT=150
KEY="$DEEPSEEK_API_KEY"

echo "########## T1-1.5B 鎺ч暱搴﹂噸璇勬祴锛圕ONCISE 150瀛楋級$(date) ##########"
for G in A B C; do
  echo "=== [T1-1.5b $G] 璇勬祴 $(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_1.5b_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_1.5b_concise_$G.json
done
tar czf t1_1.5b_results.tar.gz results/recipe_1.5b_concise_*.json
echo "===== T1-1.5B 鎺ч暱搴﹀畬鎴?$(date) ====="

