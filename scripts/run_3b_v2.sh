#!/bin/bash
# 3B菜谱版2（M分层锁文体）
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# ============================================================
# 鐗?锛?B 鑿滆氨 M 鍒嗗眰锛堥攣鏂囦綋锛塴ow/mid/high 鈥斺€?鍘熷 + 鎺ч暱搴?鍙岃瘎娴?# 鐩殑锛氬墺绂婚暱搴︿笌鏂囦綋鍚庯紝鍙嶉棶璐ㄩ噺 M 瀵?Y 鏄惁鏈夊共鍑€鐨勫崟璋冨洜鏋?# 鍓嶇疆锛氬凡璺?t4_gen_grade_locked.py锛圓PI浠诲姟锛孏PU璁粌鏃跺苟琛岃窇锛夛紝杈撳嚭
#       results/t4_cq_graded_locked.json
# 鐢ㄦ硶锛氬厛 export key锛屽啀
#   nohup ITER=5 GENB=16 LIMIT=150 bash run_3b_v2.sh > run_3b_v2.log 2>&1 &
# 鎭掓簮浜戜笉鑳藉叧鏈猴紝璺戝畬绔嬪埢涓嬭浇 tar.gz銆?# ============================================================
set -e
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

SIZE=3b
ITER="${ITER:-5}"
GENB="${GENB:-16}"
LR="${LR:-5e-6}"
LIMIT="${LIMIT:-150}"
MA_CACHE="${MA_CACHE:-cache/recipe_ma_cache.json}"
TEST=data/recipe/qa_pairs_clean/test_mve.json
TRAIN=data/recipe/qa_pairs_clean/train.json
JUDGES=configs/judges.yaml

echo "########## STEP0: 閿佹枃浣撳€欓€夊垎灞?$(date) ##########"
if [ ! -f "results/t4_cq_graded_locked.json" ]; then
  echo "缂?results/t4_cq_graded_locked.json锛岃鍏堣窇 t4_gen_grade_locked.py锛圓PI浠诲姟,鍙笌鐗?璁粌骞惰锛?; exit 1
fi
cp results/t4_cq_graded_locked.json results/t4_cq_graded.json
python t4_tertile_split.py
grep -E "^low妗^mid妗^high妗? results/t4_tertile_split.txt

echo "########## STEP1: self_play 涓夋。 ($SIZE B缁? ${ITER}杞? lr${LR}+浣欏鸡) $(date) ##########"
for T in low mid high; do
  echo "=== [t4-3b $T] self_play $(date +%H:%M:%S) ==="
  python -m src.training.self_play --sft-model models/recipe_sft_$SIZE \
    --train-data data/recipe/qa_pairs_clean/t4_$T.json --domain recipe --group B \
    --iterations "$ITER" --lr "$LR" --lr-schedule cosine --batch-size 4 \
    --train-limit 500 --gen-batch "$GENB" --ma-cache "$MA_CACHE" \
    --api-key "$KEY" --out models/t4_${SIZE}_$T
done

echo "########## STEP2: 鍘熷璇勬祴 iter${ITER} (3瑁佸垽, n=${LIMIT}) $(date) ##########"
for T in low mid high; do
  echo "=== [t4-3b $T] 鍘熷璇勬祴 $(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain recipe --group B \
    --model models/t4_${SIZE}_$T/groupB_iter${ITER} \
    --test-data "$TEST" --limit "$LIMIT" --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/t4a_${SIZE}_$T.json
done

echo "########## STEP3: 鎺ч暱搴﹁瘎娴?(CONCISE_MODE=1, 150瀛? $(date) ##########"
for T in low mid high; do
  echo "=== [t4-3b $T] 鎺ч暱搴﹁瘎娴?$(date +%H:%M:%S) ==="
  CONCISE_MODE=1 CONCISE_LIMIT=150 \
  python -m src.eval.run_eval --domain recipe --group B \
    --model models/t4_${SIZE}_$T/groupB_iter${ITER} \
    --test-data "$TEST" --limit "$LIMIT" --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/t4a_${SIZE}_concise_$T.json
done

echo "########## STEP4: 鎵撳寘 $(date) ##########"
tar czf recipe_${SIZE}_v2_results.tar.gz \
  results/t4a_${SIZE}_*.json results/t4_tertile_split.txt
echo "===== 鐗? 瀹屾垚 $(date)锛佺珛鍗充笅杞?recipe_${SIZE}_v2_results.tar.gz ====="
echo "鏈湴鍒嗘瀽锛氫笁妗?Y 閰嶅妫€楠?+ 杩炵画 M-Y 鍥炲綊 + 璺ㄨ鍒ょǔ鍋ユ€?

