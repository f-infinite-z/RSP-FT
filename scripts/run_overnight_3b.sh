#!/bin/bash
# 3B整夜连跑队列
# 开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
# ============================================================
# 3B 鏁村杩炶窇闃熷垪锛堟亽婧愪簯涓嶈兘鍏虫満锛屼竴娆℃寕璧锋Θ骞叉暣澶滐級
# 椤哄簭锛氱増1 鈫?鐗?鍊欓€?閿佹枃浣? 鈫?鐗? 鈫?娉曞緥QA鐢熸垚
# 姣忔澶辫触涓嶄腑鏂悗缁紙|| true锛夛紝鍚勮嚜鏈夋柇鐐圭画浼?鎵撳寘
# 鐢ㄦ硶锛氬厛 export key锛屽啀
#   export GENB=24
#   nohup bash run_overnight_3b.sh > overnight_3b.log 2>&1 &
#   tail -f overnight_3b.log
# 鏃╀笂鍥炴潵锛歭s *_results.tar.gz  +  妫€鏌?data/law/qa_pairs/all.json
# ============================================================
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?请设置DeepSeek API Key}
export ARK_API_KEY=${ARK_API_KEY:?请设置豆包API Key}
KEY="$DEEPSEEK_API_KEY"
export GENB="${GENB:-24}"

echo "########################################################"
echo "###### 鏁村闃熷垪鍚姩 $(date) ######"
echo "########################################################"

echo ""
echo "===== [1/4] 鐗? A/B/C + 鍙岃瘎娴?$(date) ====="
bash run_3b_v1.sh || echo "!!! 鐗? 寮傚父锛岀户缁悗缁?$(date)"

echo ""
echo "===== [2/4] 鐗? 閿佹枃浣撳€欓€夌敓鎴?API) $(date) ====="
python t4_gen_grade_locked.py --limit 500 --api-key "$KEY" \
    --judges-config configs/judges.yaml --workers 20 \
    --out results/t4_cq_graded_locked.json || echo "!!! 鍊欓€夌敓鎴愬紓甯?$(date)"

echo ""
echo "===== [3/4] 鐗? M鍒嗗眰閿佹枃浣?low/mid/high + 鍙岃瘎娴?$(date) ====="
ITER=5 GENB="$GENB" LIMIT=150 bash run_3b_v2.sh || echo "!!! 鐗? 寮傚父 $(date)"

echo ""
echo "===== [4/4] 娉曞緥QA鐢熸垚(绾疉PI,涓嶅崰GPU) $(date) ====="
python gen_law.py --api-key "$KEY" --types-per-tiao 3 --workers 15 \
    --out data/law/qa_pairs/all.json || echo "!!! 娉曞緥QA寮傚父 $(date)"

echo ""
echo "########################################################"
echo "###### 鏁村闃熷垪鍏ㄩ儴瀹屾垚 $(date) ######"
echo "###### 浜у嚭锛歳ecipe_3b_v1_results.tar.gz / recipe_3b_v2_results.tar.gz"
echo "######       data/law/qa_pairs/all.json"
echo "###### 璇峰叏閮ㄤ笅杞藉悗鍐嶈€冭檻閲婃斁瀹炰緥锛堟亽婧愪簯鍏虫満娓呮暟鎹級"
echo "########################################################"

