#!/bin/bash
# 3B菜谱版1（原始+控长度双评测）
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# ============================================================
# 鐗?锛?B 鑿滆氨 A/B/C 鈥斺€?鍘熷 + 鎺ч暱搴?鍙岃瘎娴嬶紙5杞紝瀵归綈1.5B锛?# 鐩殑锛氬閲忔槸鍚︽敼鍙?A>B>C 鏍煎眬锛?B 鎺ч暱搴﹀悗 B/C 鑳藉惁杩藉钩/鍙嶈秴 A
# 娉細config 宸插榻?1.5B 涓诲疄楠岋紙iter5/lr5e-6/cosine锛夛紝鑴氭湰涓嶈鐩栥€?# 鐢ㄦ硶锛氬厛 export key锛屽啀
#   nohup GENB=24 bash run_3b_v1.sh > run_3b_v1.log 2>&1 &
# 鎭掓簮浜戜笉鑳藉叧鏈猴紙鍏虫満娓呮暟鎹級锛岃窇瀹屾暣涓増1绔嬪埢涓嬭浇 tar.gz銆?# ============================================================
set -e
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

SIZE=3b
ITER=5                                 # config 宸插浐瀹?杞?GENB="${GENB:-16}"                     # A100-40G 鍙牎鍑嗗埌24~32
LIMIT="${LIMIT:-100}"
MA_CACHE="${MA_CACHE:-cache/recipe_ma_cache.json}"
TEST=data/recipe/qa_pairs_clean/test_mve.json
TRAIN=data/recipe/qa_pairs_clean/train.json
JUDGES=configs/judges.yaml

echo "########## STEP0: SFT 鍩虹嚎 (Qwen2.5-3B) $(date) ##########"
if [ ! -d "models/recipe_sft_$SIZE" ]; then
  python -m src.training.base_sft --config configs/sft_recipe_$SIZE.yaml
else
  echo "宸插瓨鍦?models/recipe_sft_$SIZE锛岃烦杩?SFT"
fi

echo "########## STEP1: self_play A/B/C ($SIZE, 500鏉? ${ITER}杞? 瀵归綈1.5B) $(date) ##########"
for G in A B C; do
  echo "=== [3b $G] self_play $(date +%H:%M:%S) ==="
  python -m src.training.self_play --config configs/recipe_${SIZE}_group$G.yaml \
    --api-key "$KEY" --train-limit 500 --gen-batch "$GENB" \
    ${MA_CACHE:+--ma-cache "$MA_CACHE"}
done

echo "########## STEP2: 鍘熷璇勬祴 iter${ITER} (3瑁佸垽) $(date) ##########"
for G in A B C; do
  echo "=== [3b $G] 鍘熷璇勬祴 $(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_${SIZE}_group$G/group${G}_iter${ITER} \
    --test-data "$TEST" --limit "$LIMIT" --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/recipe_${SIZE}_$G.json
done
python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_${SIZE}_A.json results/recipe_${SIZE}_B.json results/recipe_${SIZE}_C.json \
  --judges-config "$JUDGES" --out results/recipe_${SIZE}_compare.json

echo "########## STEP3: 鎺ч暱搴﹁瘎娴?(CONCISE_MODE=1, 150瀛? $(date) ##########"
for G in A B C; do
  echo "=== [3b $G] 鎺ч暱搴﹁瘎娴?$(date +%H:%M:%S) ==="
  CONCISE_MODE=1 CONCISE_LIMIT=150 \
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_${SIZE}_group$G/group${G}_iter${ITER} \
    --test-data "$TEST" --limit "$LIMIT" --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/recipe_${SIZE}_concise_$G.json
done
CONCISE_MODE=1 CONCISE_LIMIT=150 \
python -m src.eval.run_eval --domain recipe --compare \
  results/recipe_${SIZE}_concise_A.json results/recipe_${SIZE}_concise_B.json results/recipe_${SIZE}_concise_C.json \
  --judges-config "$JUDGES" --out results/recipe_${SIZE}_concise_compare.json

echo "########## STEP4: 涓粙鍒嗘瀽 (鍘熷) $(date) ##########"
python -m src.eval.run_eval --aggregate \
  results/recipe_${SIZE}_A.json results/recipe_${SIZE}_B.json results/recipe_${SIZE}_C.json \
  --out results/recipe_${SIZE}_mediation.txt

echo "########## STEP5: 鎵撳寘 $(date) ##########"
tar czf recipe_${SIZE}_v1_results.tar.gz \
  results/recipe_${SIZE}_*.json results/recipe_${SIZE}_mediation.txt
echo "===== 鐗? 瀹屾垚 $(date)锛佺珛鍗充笅杞?recipe_${SIZE}_v1_results.tar.gz ====="
echo "璇风珛鍗充笅杞界粨鏋滐紝鐒跺悗纭鏄惁椤烘鍚姩鐗?锛堟亽婧愪簯涓嶈兘鍏筹級銆?

