#!/bin/bash
# 全量实验（两领域×A/B/C）
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# 鍏ㄩ噺瀹為獙锛堣蛋鍚?锛歭r5璇婃柇纭鏂规硶鏈夋晥鍚庯級
# 涓ら鍩?脳 A/B/C锛岃缁?500 + 璇勬祴150锛宭r=5e-6+浣欏鸡閫€鐏?杞紝鎵归噺鐢熸垚+棰勭敓鎴愮紦瀛?# 鐢ㄦ硶锛氬厛 export 涓変釜key锛屽啀 nohup bash run_full.sh > full.log 2>&1 &
# 鍓嶆彁锛氬厛璺戦鐢熸垚缂撳瓨锛堣剼鏈唴 STEP0 鑷姩璺戯級
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"

echo "########## STEP0: 棰勭敓鎴?500鏉＄紦瀛橈紙涓ら鍩燂級##########"
for DOM in poetry recipe; do
  if [ "$DOM" = "poetry" ]; then TRAIN=data/poetry/qa_pairs/train.json; else TRAIN=data/recipe/qa_pairs_clean/train.json; fi
  if [ ! -f "cache/${DOM}_full_cache.json" ] || [ "$(python -c "import json;print(len(json.load(open('cache/${DOM}_full_cache.json'))))" 2>/dev/null || echo 0)" -lt 4000 ]; then
    echo "=== 棰勭敓鎴?$DOM 1500鏉＄紦瀛?==="
    python pregen_counter_answers.py --domain $DOM \
      --train-data $TRAIN --limit 1500 \
      --api-key "$KEY" --workers 30 \
      --out cache/${DOM}_full_cache.json
  else
    echo "$DOM 缂撳瓨宸插瓨鍦紝璺宠繃"
  fi
done

# 鍏ㄩ噺鐢ㄧ殑 config锛歵rain-limit 1500锛岃瘎娴?150銆傝繖閲岀敤鍛戒护琛岃鐩?lr5 config 鐨勮矾寰勩€?run_domain () {
  local DOM=$1 TRAIN=$2 TEST=$3 CACHE=$4
  echo "########## $DOM 鍏ㄩ噺璁粌锛?500鏉?lr5+浣欏鸡,5杞?鎵归噺锛?#########"
  for G in A B C; do
    echo "=== [$DOM-full $G] self_play ==="
    python -m src.training.self_play --config configs/${DOM}_lr5_group$G.yaml \
      --train-data $TRAIN --api-key "$KEY" --train-limit 1500 \
      --ma-cache $CACHE --gen-batch 48
  done
  echo "########## $DOM 鍏ㄩ噺璇勬祴锛?50鏉?3瑁佸垽,iter5锛?#########"
  for G in A B C; do
    python -m src.eval.run_eval --domain $DOM --group $G \
      --model models/${DOM}_lr5_group$G/group${G}_iter5 \
      --test-data $TEST --limit 150 \
      --train-data $TRAIN \
      --judges-config configs/judges.yaml --api-key "$KEY" \
      --out results/${DOM}_full_$G.json
  done
  python -m src.eval.run_eval --domain $DOM --compare \
    results/${DOM}_full_A.json results/${DOM}_full_B.json results/${DOM}_full_C.json \
    --judges-config configs/judges.yaml --out results/${DOM}_full_compare.json
  python -m src.eval.run_eval --aggregate \
    results/${DOM}_full_A.json results/${DOM}_full_B.json results/${DOM}_full_C.json \
    --out results/${DOM}_full_mediation.txt
}

run_domain poetry data/poetry/qa_pairs/train.json data/poetry/qa_pairs/test.json cache/poetry_full_cache.json
run_domain recipe data/recipe/qa_pairs_clean/train.json data/recipe/qa_pairs_clean/test.json cache/recipe_full_cache.json

tar czf full_results.tar.gz results/poetry_full_*.json results/poetry_full_mediation.txt results/recipe_full_*.json results/recipe_full_mediation.txt
echo "===== 鍏ㄩ噺瀹屾垚锛佷笅杞?full_results.tar.gz ====="

