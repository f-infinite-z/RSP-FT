# 复现指南 (REPRODUCE)

本文档提供从零复现 RSP-FT 实验的完整步骤。

## 0. 硬件与环境

- **GPU**：单卡即可。0.5B 模型 ~10GB 显存；1.5B ~10GB（batch=2）；3B 需 24GB+。
- **环境**：Python 3.11，CUDA 12.x。
```bash
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内加速（可选）
export HF_HUB_DISABLE_XET=1                # 用hf-mirror时必加：xet协议与镜像不兼容
```

## 1. 数据准备

数据已包含在 `data/`：
- `data/poetry/qa_pairs/`：古诗词 QA（train/val/test）
- `data/recipe/qa_pairs_clean/`：菜谱 QA（清洗后）

如需重新生成，见 `data/README.md`。

## 2. 裁判配置

```bash
cp configs/judges.yaml.example configs/judges.yaml
# 编辑填入三家 API key，或用环境变量：
export DEEPSEEK_API_KEY=...
export ARK_API_KEY=...          # 豆包（火山方舟）
export DASHSCOPE_API_KEY=...    # 千问（阿里百炼）
```

## 3. SFT 领域基线

```bash
python -m src.training.base_sft --config configs/sft_recipe.yaml
python -m src.training.base_sft --config configs/sft_poetry.yaml
```
产出 `models/recipe_sft/`、`models/poetry_sft/`。

## 4. 预生成 M_A 补充缓存（强烈推荐，大幅加速）

训练时 B/C 组需 M_A 对反问给出补充。预生成缓存可避免训练时串行等待 API：

```bash
mkdir -p cache
# 先小批测并发（确认不触发限流）
python pregen_counter_answers.py --domain recipe \
    --train-data data/recipe/qa_pairs_clean/train.json --limit 50 \
    --api-key $DEEPSEEK_API_KEY --workers 10 --out cache/recipe_ma_cache.json
# 正式生成（并发可加大）
python pregen_counter_answers.py --domain recipe \
    --train-data data/recipe/qa_pairs_clean/train.json --limit 500 \
    --api-key $DEEPSEEK_API_KEY --workers 30 --out cache/recipe_ma_cache.json
```

## 5. 三组自博弈训练

```bash
export MA_CACHE=cache/recipe_ma_cache.json
for G in A B C; do
  python -m src.training.self_play --config configs/recipe_group$G.yaml \
    --api-key $DEEPSEEK_API_KEY --train-limit 500 \
    --ma-cache $MA_CACHE --gen-batch 48
done
```

**关键参数**：
- `--train-limit N`：限训练样本量（可复现，取前 N 条）
- `--ma-cache`：M_A 补充缓存路径（命中则零 API 等待）
- `--gen-batch`：批量生成大小（吃满 GPU；1.5B 用 16，0.5B 用 48）
- config 中 `lr-schedule: cosine`：余弦退火（应对大 lr 震荡）

## 6. 三裁判评测

```bash
for G in A B C; do
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_group$G/group${G}_iter5 \
    --test-data data/recipe/qa_pairs_clean/test.json --limit 100 \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --judges-config configs/judges.yaml --api-key $DEEPSEEK_API_KEY \
    --out results/recipe_$G.json
done
# 对比+CoT 评测（一次调用比较三组）
python -m src.eval.run_eval --domain recipe --compare \
    results/recipe_A.json results/recipe_B.json results/recipe_C.json \
    --judges-config configs/judges.yaml --out results/recipe_compare.json
```

## 7. 分析与出图

```bash
# 配对检验 + Cohen's d（判断组间差异是否显著）
python stat_test.py --domains recipe poetry

# 混淆探查：长度效应 + 维度分解 + 连续 M-Y 回归（核心）
python probe_confounders.py --prefix recipe

# 中介分析（Hayes Model4）
python -m src.eval.run_eval --aggregate \
    results/recipe_A.json results/recipe_B.json results/recipe_C.json \
    --out results/recipe_mediation.txt

# 核心结果图（M-Y散点/训练强度曲线/长度/四维分解）
python make_paper_figs.py
```

## 8. 关键分析脚本说明

| 脚本 | 作用 |
|------|------|
| `stat_test.py` | 组间配对检验、效应量、评分区分度 |
| `probe_confounders.py` | 长度效应、维度崩溃定位、**连续 M-Y 回归**、裁判分歧 |
| `probe_mechanism.py` | 长度膨胀、格式污染（重复度）检验 |
| `judge_perspective.py` | 裁判×组别交叉表（评价视角调节） |
| `make_paper_figs.py` | 论文核心图 |
| `src/analysis/mediation.py` | Hayes Model4 链式中介（纯 numpy） |

## 9. 训练强度梯度（可选，复现"训练强度调节"发现）

```bash
# 扫描不同 lr，观察 M-Y 相关如何随训练强度变化
for LR in 2e-6 5e-6 1e-5; do
  # 用对应 lr 的 config，跑 B/C 组 + 评测，再用 probe_confounders 看 r(M,Y)
done
```

## 常见问题

- **训练慢/GPU 利用率低**：确认用了 `--ma-cache`（消除 API 等待）和 `--gen-batch`（批量生成）。逐条生成会很慢。
- **CUDA OOM**：降 `batch-size`（config）和 `--gen-batch`；大模型降 `max-len`。
- **裁判 API 超时**：内置重试+退避；豆包偶发 500 会自动重试。
- **结论只看均值会误判**：务必用 `stat_test.py` 的配对检验+效应量；反问质量效应用 `probe_confounders.py` 的连续 M-Y 回归看。
