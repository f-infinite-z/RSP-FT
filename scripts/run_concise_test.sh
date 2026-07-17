#!/bin/bash
# 精简模式快验证
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 绮剧畝妯″紡蹇獙璇侊紙鏂规B锛夛細鎺у埗闀垮害鍚庯紝鍙嶉棶缁刌鏄惁鍙嶈秴A锛?# 鐢ㄧ幇鏈?MVE 妯″瀷(recipe_group{A,B,C}/group{X}_iter3) 閲嶈瘎娴嬶紝寮€鍚?CONCISE_MODE
# A/B/C 涓夌粍缁熶竴鍔犵簿绠€鎸囦护锛堝叕骞筹級锛屽姣?绮剧畝 vs 鍘熷"鐨勭粍闂存牸灞€
# 鐢ㄦ硶锛氬厛export key锛屽啀 nohup bash run_concise_test.sh > concise_test.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
export CONCISE_MODE=1
export CONCISE_LIMIT=150
KEY="$DEEPSEEK_API_KEY"

echo "########## 绮剧畝妯″紡閲嶈瘎娴嬶紙CONCISE_MODE=1, 150瀛? MVE妯″瀷iter3锛?#########"
for G in A B C; do
  echo "=== [concise $G] 璇勬祴 ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_group$G/group${G}_iter3 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_concise_$G.json
done

tar czf concise_results.tar.gz results/recipe_concise_*.json
echo "===== 绮剧畝妯″紡蹇獙璇佸畬鎴愶紒涓嬭浇 concise_results.tar.gz ====="
echo "瀵规瘮锛歳esults/recipe_concise_* (绮剧畝) vs results/recipe_* (鍘熷)"

