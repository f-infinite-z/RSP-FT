#!/bin/bash
# 3B第二阶段队列
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# ============================================================
# 绗簩闃舵闃熷垪锛堢増1 瀹屾垚鍚庡啀鎸傦紝涓嶅惈鐗?锛岄伩鍏嶉噸璺戯級
# 椤哄簭锛氶攣鏂囦綋鍊欓€夌敓鎴?宸蹭慨) -> 鐗? M鍒嗗眰 -> 娉曞緥QA鐢熸垚
# 鍓嶇疆锛氱増1 宸插畬鎴愶紙recipe_3b_v1_results.tar.gz 宸蹭骇鍑?涓嬭浇锛?# 鐢ㄦ硶锛歟xport GENB=24; nohup bash run_phase2_3b.sh > phase2_3b.log 2>&1 &
# ============================================================
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"
export GENB="${GENB:-24}"

echo "###### 绗簩闃舵闃熷垪鍚姩 $(date) ######"

echo ""
echo "===== [1/3] 閿佹枃浣撳€欓€夌敓鎴?API,宸蹭慨瀛楁暟杩囨护) $(date) ====="
python t4_gen_grade_locked.py --limit 500 --api-key "$KEY" \
    --judges-config configs/judges.yaml --workers 20 \
    --out results/t4_cq_graded_locked.json

N=$(python -c "import json;print(len(json.load(open('results/t4_cq_graded_locked.json',encoding='utf-8'))))" 2>/dev/null || echo 0)
echo "閿佹枃浣撳€欓€夎褰曟暟: $N"
if [ "$N" -lt 100 ]; then
  echo "!!! 鍊欓€夋暟<100锛岀増2鏁版嵁涓嶈冻锛岃烦杩囩増2锛岀洿鎺ユ硶寰婹A銆傝鎺掓煡 t4_gen_grade_locked.py"
else
  echo ""
  echo "===== [2/3] 鐗? M鍒嗗眰閿佹枃浣?+ 鍙岃瘎娴?$(date) ====="
  ITER=5 GENB="$GENB" LIMIT=150 bash run_3b_v2.sh || echo "!!! 鐗?寮傚父 $(date)"
fi

echo ""
echo "===== [3/3] 娉曞緥QA鐢熸垚(绾疉PI) $(date) ====="
python gen_law.py --api-key "$KEY" --types-per-tiao 3 --workers 15 \
    --out data/law/qa_pairs/all.json || echo "!!! 娉曞緥QA寮傚父 $(date)"

echo ""
echo "###### 绗簩闃舵瀹屾垚 $(date) ######"
echo "###### 浜у嚭: recipe_3b_v2_results.tar.gz / data/law/qa_pairs/all.json"

