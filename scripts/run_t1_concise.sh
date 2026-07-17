#!/bin/bash
# T1精简模式(0.5B)
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# T1 绮剧畝妯″紡楠岃瘉锛堟柟妗圔锛夛細鎺у埗闀垮害鍚庯紝鍙嶉棶缁刌鏄惁鍙嶈秴A锛? 鎻愰€熷箙搴?# 鐢?0.5B lr5 妯″瀷(recipe_lr5_group{A,B,C}/group{X}_iter5锛岃缁冨厖鍒嗙増锛岄暱搴﹁啫鑳€鏄庢樉)閲嶈瘎娴?# 寮€鍚?CONCISE_MODE(鍏堟⒊鐞?150瀛楀唴)锛孉/B/C涓夌粍缁熶竴(鍏钩)
# 瀵规瘮锛歳ecipe_concise_*(绮剧畝) vs recipe_lr5_*(鍘熷) 鐨?Y銆侀暱搴︺€佽€楁椂
# 鐢ㄦ硶锛氬厛export key锛屽啀 nohup bash run_t1_concise.sh > t1_concise.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
export CONCISE_MODE=1
export CONCISE_LIMIT=150
KEY="$DEEPSEEK_API_KEY"

echo "########## T1 绮剧畝妯″紡閲嶈瘎娴嬶紙0.5B lr5 iter5, CONCISE_MODE=1, 150瀛楋級##########"
echo "寮€濮嬫椂闂? $(date)"
for G in A B C; do
  echo "=== [T1-concise $G] 璇勬祴 寮€濮?$(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_lr5_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_concise_$G.json
  echo "=== [T1-concise $G] 瀹屾垚 $(date +%H:%M:%S) ==="
done

tar czf t1_concise_results.tar.gz results/recipe_concise_*.json
echo "===== T1 绮剧畝楠岃瘉瀹屾垚 $(date)锛佷笅杞?t1_concise_results.tar.gz ====="
echo "瀵规瘮鑴氭湰(鏈湴): 绮剧畝 recipe_concise_* vs 鍘熷 recipe_lr5_*"
echo "鐪嬬偣锛氣憼鎺ч暱搴﹀悗A/B/C鐨刌鏄惁鍙樺寲(B/C鏄惁鍙嶈秴A) 鈶＄瓟妗堥暱搴?搴攡150瀛? 鈶㈣瘎娴嬭€楁椂(搴旀瘮鍘熷蹇?"

