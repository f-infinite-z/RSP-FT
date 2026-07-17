#!/bin/bash
# 大模型重测菜谱(支持1.5B/3B)
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 鏂规2锛氭崲澶фā鍨嬮噸娴嬭彍璋憋紝妫€楠?鑳藉姏澶╄姳鏉?鏄惁涓鸿礋缁撴灉涓诲洜
# 鐢ㄦ硶锛歋IZE=1.5b ./run_recipe_bigmodel.sh   鎴?  SIZE=3b ...
#   鍏?export key锛涘缓璁?nohup SIZE=1.5b ./run_recipe_bigmodel.sh > recipe_1.5b.log 2>&1 &
# 鍓嶆彁锛氬厛璺戣 size 鐨?SFT锛堣鑴氭湰鍐?STEP0锛夛紝鎴栧凡瀛樺湪 models/recipe_sft_$SIZE
# 娉ㄦ剰锛?B 鍦?4090-24G 涓婅窇锛宐atch 宸插湪 config 闄嶅埌 4锛涚敤瀹岀珛鍗冲叧鏈?set -e
SIZE="${SIZE:?璇锋寚瀹?SIZE=1.5b 鎴?SIZE=3b}"
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

echo "########## STEP0: SFT 鍩虹嚎 (Qwen2.5-$SIZE) ##########"
if [ ! -d "models/recipe_sft_$SIZE" ]; then
  python -m src.training.base_sft --config configs/sft_recipe_$SIZE.yaml
else
  echo "宸插瓨鍦?models/recipe_sft_$SIZE锛岃烦杩?SFT"
fi

echo "########## STEP1: 涓夌粍 self_play ($SIZE, 500鏉? 5杞? lr5e-6+浣欏鸡+鎵归噺) ##########"
for G in A B C; do
  echo "=== [recipe-$SIZE $G] self_play ==="
  python -m src.training.self_play --config configs/recipe_${SIZE}_group$G.yaml \
    --api-key "$KEY" --train-limit 500 --gen-batch 16 ${MA_CACHE:+--ma-cache $MA_CACHE}
done

echo "########## STEP2: 璇勬祴 iter5 (3瑁佸垽) ##########"
for G in A B C; do
  echo "=== [recipe-$SIZE $G] 璇勬祴 iter5 ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_${SIZE}_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_${SIZE}_$G.json
done

python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_${SIZE}_A.json results/recipe_${SIZE}_B.json results/recipe_${SIZE}_C.json \
  --judges-config configs/judges.yaml --out results/recipe_${SIZE}_compare.json

echo "########## STEP3: 涓粙鍒嗘瀽 ##########"
python -m src.eval.run_eval --aggregate \
  results/recipe_${SIZE}_A.json results/recipe_${SIZE}_B.json results/recipe_${SIZE}_C.json \
  --out results/recipe_${SIZE}_mediation.txt

tar czf recipe_${SIZE}_results.tar.gz results/recipe_${SIZE}_*.json results/recipe_${SIZE}_mediation.txt
echo "===== $SIZE 閲嶆祴瀹屾垚锛佷笅杞?recipe_${SIZE}_results.tar.gz锛岀劧鍚庛€愬叧鏈烘鎹熴€?====="

