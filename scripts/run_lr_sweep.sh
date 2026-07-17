#!/bin/bash
# 训练强度梯度扫描
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 璁粌寮哄害姊害鎵弿锛堢枒鐐?/2/6锛夛細涓嶅悓lr涓?M-Y 鐩稿叧濡備綍鍙樺寲
# 鑿滆氨锛宭r鈭坽2e-6,1e-5,2e-5}锛屽悇璺態/C缁勶紙A缁勬棤鍙嶉棶涓嶅奖鍝峂-Y锛夛紝5杞?浣欏鸡+鎵归噺+缂撳瓨
# 宸叉湁 lr=1e-6(MVE) 鍜?5e-6(lr5) 鏁版嵁锛屾湰鑴氭湰琛ラ綈姊害
# 鐢ㄦ硶锛氬厛export涓塳ey+MA_CACHE锛屽啀 nohup bash run_lr_sweep.sh > lr_sweep.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
export MA_CACHE=cache/recipe_ma_cache.json
KEY="$DEEPSEEK_API_KEY"

for TAG in 2e6 1e5 2e5; do
  echo "########## lr=$TAG 璁粌+璇勬祴 ##########"
  for G in B C; do
    echo "=== [lr$TAG $G] self_play ==="
    python -m src.training.self_play --config configs/recipe_lr${TAG}_group$G.yaml \
      --api-key "$KEY" --train-limit 500 --ma-cache "$MA_CACHE" --gen-batch 48
  done
  for G in B C; do
    echo "=== [lr$TAG $G] 璇勬祴 iter5 ==="
    python -m src.eval.run_eval --domain recipe --group $G \
      --model models/recipe_lr${TAG}_group$G/group${G}_iter5 \
      --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
      --train-data data/recipe/qa_pairs_clean/train.json \
      --judges-config configs/judges.yaml --api-key "$KEY" \
      --out results/recipe_lr${TAG}_$G.json
  done
done

tar czf lr_sweep_results.tar.gz results/recipe_lr2e6_*.json results/recipe_lr1e5_*.json results/recipe_lr2e5_*.json
echo "===== 璁粌寮哄害姊害鎵弿瀹屾垚锛佷笅杞?lr_sweep_results.tar.gz ====="

