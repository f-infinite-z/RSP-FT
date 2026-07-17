# RSP-FT：反问式自博弈微调 (Reciprocal Self-Play Fine-Tuning)

> 让垂直领域小语言模型通过"反问"学会更好地回答。

本仓库为论文《反问式自博弈微调：让垂直领域语言模型通过反问学会更好地回答》的官方代码与数据实现，提供完整的可复现实验流程。

📄 论文：[待补：期刊/arXiv 链接]
🔗 引用：见 [Citation](#citation)

---

## 简介

**RSP-FT** 在自博弈微调循环中引入"反问"作为额外训练信号：回答者在初次回答后针对自己的回答提出反问，由提问者给出仿真补充，再据此产出最终回答；初答、反问、最终答三段文本共同构成交叉熵训练信号。

我们通过 A/B/C 三组分层操纵与链式中介分析，系统考察了反问对垂直领域小模型（中华菜谱、古诗词）回答质量的影响。

### 核心发现

1. **反问质量决定回答质量**：以连续反问质量 M 为自变量，M 与回答质量 Y 在两领域、两训练配置下稳健正相关（r=0.25–0.46）。反问机制的价值载体是反问的**质量**，而非有无或强弱。
2. **三重混淆掩盖真实效应**：分组层面的"负结果"由操纵污染、长度膨胀、训练强度挤占三重混淆造成，而非反问机制本身的缺陷。
3. **反问的双刃剑**：高强度反问提升分析深度但可能损害事实准确性，净效应受领域与评价视角调节。

---

## 快速开始

### 环境

```bash
# 推荐：安装作者复现实测的锁定版本（云GPU镜像先卸载预装的 torchvision/torchaudio，避免与升级后torch冲突）
pip uninstall -y torchvision torchaudio
pip install -r requirements.lock -i https://pypi.tuna.tsinghua.edu.cn/simple
# 主要依赖：torch 2.5.1, transformers 5.14, peft 0.19, datasets, openai, numpy, matplotlib
# requirements.txt 为宽松下限，仅供参考；版本组合以 requirements.lock 为准
```

### 配置裁判 API

复制 `configs/judges.yaml.example` 为 `configs/judges.yaml`，填入你的 API Key（或用环境变量）：

```bash
export DEEPSEEK_API_KEY=your_key      # DeepSeek (M_A补充 + 裁判)
export ARK_API_KEY=your_key           # 豆包 Doubao (裁判)
export DASHSCOPE_API_KEY=your_key     # 通义千问 Qwen (裁判)
```

### 复现主要实验

```bash
# 1. SFT 领域基线
python -m src.training.base_sft --config configs/sft_recipe.yaml

# 2. (可选,加速)预生成 M_A 补充缓存
python pregen_counter_answers.py --domain recipe \
    --train-data data/recipe/qa_pairs_clean/train.json --limit 500 \
    --api-key $DEEPSEEK_API_KEY --workers 30 --out cache/recipe_ma_cache.json

# 3. 三组自博弈训练（A/B/C）
for G in A B C; do
  python -m src.training.self_play --config configs/recipe_group$G.yaml \
    --api-key $DEEPSEEK_API_KEY --ma-cache cache/recipe_ma_cache.json --gen-batch 48
done

# 4. 三裁判评测
for G in A B C; do
  python -m src.eval.run_eval --domain recipe --group $G \
    --model models/recipe_group$G/group${G}_iter3 \
    --test-data data/recipe/qa_pairs_clean/test.json \
    --judges-config configs/judges.yaml --api-key $DEEPSEEK_API_KEY \
    --out results/recipe_$G.json
done

# 5. 统计分析与出图
python stat_test.py                        # 配对检验 + 效应量
python probe_confounders.py --prefix recipe # 长度/维度/M-Y回归
python make_paper_figs.py                  # 核心结果图
```

详细复现步骤见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

---

## 目录结构

```
RSP-FT/
├── README.md
├── requirements.txt
├── src/
│   ├── training/         # base_sft, self_play, dialogue, counter_answer
│   ├── counter_question/ # golden_questions, templates
│   ├── judge/            # llm_judge, judge_factory (三裁判)
│   ├── eval/             # run_eval (生成+评测+对比CoT)
│   ├── analysis/         # mediation (Hayes Model4中介)
│   └── data/             # 数据处理
├── configs/              # 实验配置 + judges.yaml.example
├── scripts/              # pregen / stat_test / probe / make_figs
├── data/                 # 数据集（见 data/README）
└── docs/
    ├── REPRODUCE.md      # 复现指南
    └── LESSONS.md        # 方法论经验与踩坑（可复现性核心）
```

---

## 方法概览

| 组 | 反问强度 X | 说明 |
|----|-----------|------|
| A  | 0 | 单向问答（无反问基线） |
| B  | 1 | 往复对话 + 数据集普通反问 |
| C  | 2 | 往复对话 + 黄金三问（高质反问模板） |

- **裁判**：DeepSeek-V4 / Doubao-Seed-Lite / Qwen 三裁判交叉评分，1–10 分制，四维（事实性/完备度/逻辑/深度）
- **中介分析**：Hayes PROCESS Model 4，X→M→Y，Bootstrap 置信区间
- **统计**：配对检验 + Cohen's d 效应量，避免仅凭均值下结论

---

## 数据

- **古诗词**：基于开源 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，大模型生成 QA。
- **中华菜谱**：大模型两步生成（菜名清单→逐菜 QA），两裁判自动清洗。

详见 [`data/README.md`](data/README.md)。

---

## 方法论经验（推荐阅读）

我们在 [`docs/LESSONS.md`](docs/LESSONS.md) 中诚实记录了实验中发现并修正的方法论问题（操纵检验失败、长度混淆、训练强度调节等），供后续研究者避坑与借鉴。这也是本工作可复现性与严谨性的重要组成。

---

## Citation

```bibtex
@article{rspft2026,
  title   = {反问式自博弈微调：让垂直领域语言模型通过反问学会更好地回答},
  author  = {[待补]},
  journal = {[待补]},
  year    = {2026}
}
```

## License

[待定：MIT / Apache-2.0]

## 致谢与合作

欢迎对反问式训练机制感兴趣的研究者交流合作（见论文"未来工作"列出的方向）。Issue / PR welcome。
