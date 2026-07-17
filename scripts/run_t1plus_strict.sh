#!/bin/bash
# T1-plus扣题防编造
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# T1-plus 鎵ｉ闃茬紪閫犵増锛欳ONCISE_STRICT=1锛堟帶闀垮害+绱ф墸闂+绂佹缂栭€?鍫嗙爩/鍋忛锛?# 鐢ㄩ€旓細鑻1(绾帶闀垮害)浠嶄笉鑳借B/C鏀瑰杽锛岀敤鏈剼鏈獙璇?鎵ｉ闃茬紪閫?鑳藉惁鏀瑰杽
#       閫掕繘璁捐锛氱函鎺ч暱搴?T1) 鈫?鎺ч暱搴?鎵ｉ(T1-plus)锛屽尯鍒?闀垮害闂"vs"缂栭€犻棶棰?
# 鐢?0.5B lr5 iter5 妯″瀷閲嶈瘎娴嬶紝A/B/C涓夌粍缁熶竴(鍏钩)
# 鐢ㄦ硶锛歯ohup bash run_t1plus_strict.sh > t1plus_strict.log 2>&1 &
set -e
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
export CONCISE_STRICT=1
export CONCISE_LIMIT=150
KEY="$DEEPSEEK_API_KEY"

echo "########## T1-plus 鎵ｉ闃茬紪閫犻噸璇勬祴锛?.5B lr5 iter5, CONCISE_STRICT锛?(date) ##########"
for G in A B C; do
  echo "=== [T1plus $G] 璇勬祴 $(date +%H:%M:%S) ==="
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_lr5_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test_mve.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key "$KEY" \
    --out results/recipe_strict_$G.json
done
tar czf t1plus_results.tar.gz results/recipe_strict_*.json
echo "===== T1-plus 瀹屾垚 $(date)锛佷笅杞?t1plus_results.tar.gz ====="
echo "涓夋柟瀵规瘮(鏈湴analyze): 鍘熷recipe_lr5 vs 鎺ч暱搴ecipe_concise vs 鎺ч暱搴?鎵ｉrecipe_strict"

