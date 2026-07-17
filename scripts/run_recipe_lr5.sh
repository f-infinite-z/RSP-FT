#!/bin/bash
# 菜谱调lr重测(lr=5e-6+余弦5轮)
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 鏂规1锛氳皟澶?lr(1e-6鈫?e-6) + 鍔犺疆鏁?3鈫?) 閲嶆祴鑿滆氨锛屾帓闄?璁粌涓嶈冻"娣锋穯
# 鍓嶆彁锛氳彍璋辫礋缁撴灉鏃舵墽琛岋紱浠嶇敤 0.5B/3090Ti锛屾渶鐪?# 鐢ㄦ硶锛氬厛 export key锛屽啀 nohup ./run_recipe_lr5.sh > recipe_lr5.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

echo "########## 鑿滆氨 璋僱r閲嶆祴 (lr=5e-6, 5杞? 500鏉? ##########"
for G in A B C; do
  echo "=== [recipe-lr5 $G] self_play ==="
  python -m src.training.self_play --config configs/recipe_lr5_group$G.yaml \
    --api-key "$KEY" --train-limit 500 ${MA_CACHE:+--ma-cache $MA_CACHE}
done

echo "########## 璇勬祴 iter5 (3瑁佸垽) ##########"
for G in A B C; do
  echo "=== [recipe-lr5 $G] 璇勬祴 iter5 ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_lr5_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_lr5_$G.json
done

python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_lr5_A.json results/recipe_lr5_B.json results/recipe_lr5_C.json \
  --judges-config configs/judges.yaml --out results/recipe_lr5_compare.json

echo "########## 涓粙鍒嗘瀽 ##########"
python -m src.eval.run_eval --aggregate \
  results/recipe_lr5_A.json results/recipe_lr5_B.json results/recipe_lr5_C.json \
  --out results/recipe_lr5_mediation.txt

tar czf recipe_lr5_results.tar.gz results/recipe_lr5_*.json results/recipe_lr5_mediation.txt
echo "===== 璋僱r閲嶆祴瀹屾垚锛佷笅杞?recipe_lr5_results.tar.gz ====="

