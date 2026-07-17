#!/bin/bash
# 法律领域A/B组实验
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# ============================================================
# 娉曞緥(姘戞硶鍏?棰嗗煙 A/B 涓ょ粍 鈥斺€?0.5B锛岃法棰嗗煙瀵圭収楠岃瘉 M-Y
# 涓庤彍璋?璇楄瘝瀵归綈锛歁VE 500/100锛宭r1e-6锛?杞紝涓嶈窇C缁?涓嶇敤榛勯噾涓夐棶)
# 鐢ㄦ硶锛氬厛 export key锛屽啀 nohup bash run_law.sh > run_law.log 2>&1 &
# 0.5B 寰堝揩锛屾亽婧愪簯3090瓒冲銆傝窇瀹屼笅杞?law_results.tar.gz銆?# ============================================================
set -e
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"
TEST=data/law/qa_pairs_clean/test_mve.json
TRAIN=data/law/qa_pairs_clean/train_mve.json
JUDGES=configs/judges.yaml
MA_CACHE=cache/law_ma_cache.json

echo "########## STEP0: SFT 鍩虹嚎 (law, 0.5B) $(date) ##########"
if [ ! -d "models/law_sft" ]; then
  python -m src.training.base_sft --config configs/sft_law.yaml
else
  echo "宸插瓨鍦?models/law_sft锛岃烦杩?
fi

echo "########## STEP1: self_play A/B (law, 500鏉? 3杞? $(date) ##########"
for G in A B; do
  echo "=== [law $G] self_play $(date +%H:%M:%S) ==="
  python -m src.training.self_play --config configs/law_group$G.yaml \
    --api-key "$KEY" ${MA_CACHE:+--ma-cache "$MA_CACHE"}
done

echo "########## STEP2: 鍘熷璇勬祴 iter3 (3瑁佸垽) $(date) ##########"
for G in A B; do
  echo "=== [law $G] 璇勬祴 $(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain law --group $G \
    --model models/law_group$G/group${G}_iter3 \
    --test-data "$TEST" --limit 100 --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/law_$G.json
done
python -m src.eval.run_eval --domain law --compare \
  results/law_A.json results/law_B.json \
  --judges-config "$JUDGES" --out results/law_compare.json 2>/dev/null || echo "(compare璺宠繃,A/B涓ょ粍compare鍙€?"

echo "########## STEP3: 鎺ч暱搴﹁瘎娴?(CONCISE_MODE=1) $(date) ##########"
for G in A B; do
  CONCISE_MODE=1 CONCISE_LIMIT=150 \
  python -m src.eval.run_eval --domain law --group $G \
    --model models/law_group$G/group${G}_iter3 \
    --test-data "$TEST" --limit 100 --train-data "$TRAIN" \
    --judges-config "$JUDGES" --api-key "$KEY" \
    --out results/law_concise_$G.json
done

echo "########## STEP4: 鎵撳寘 $(date) ##########"
tar czf law_results.tar.gz results/law_*.json
echo "===== 娉曞緥瀹為獙瀹屾垚 $(date)锛佷笅杞?law_results.tar.gz ====="
echo "鏈湴鍒嗘瀽锛欱缁勮繛缁?M-Y 鐩稿叧锛堥獙璺ㄩ鍩燂級锛孉/B閰嶅妫€楠岋紝鎺ч暱搴﹀姣?

