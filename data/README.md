# 数据集说明

## 目录结构（重要）

```
data/
├── examples/                      # ⚠️ 仅为格式样例（各5条），供快速查看字段结构，勿用于训练
│   ├── poetry_qa_sample.json
│   └── recipe_qa_sample.json
├── poetry/qa_pairs/               # 古诗词完整数据集（train 4999 / val 500 / test 500）
├── recipe/qa_pairs_clean/         # 菜谱清洗版完整数据集（train 1697 / val 140 / test 160）
└── law/qa_pairs_clean/            # 法律完整数据集（train 2409 / val 200 / test 300，另含 *_mve 子集）
```

**训练/评测请直接使用上述完整数据集路径（configs/ 与复现指南中的默认路径即指向它们）。**

跑实验前建议先核对数据量级，避免误用样例：

```bash
python -c "import json; print(len(json.load(open('data/recipe/qa_pairs_clean/train.json',encoding='utf-8'))))"
# 预期 1697；若只有个位数说明拿错了文件
```

## 古诗词 QA

- **来源**：基于开源项目 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（39万+ 首诗词），用大模型批量生成问答对。
- **规模**：train 4999 / val 500 / test 500。
- **题型**：情感主旨、意象象征、修辞手法、结构脉络、朝代背景等多种。
- **字段**：`question` / `answer` / `qa_type` / `counter_question`（普通反问，供 B 组）/ `counter_answer`。
- **上下文注入**：为避免"问题脱离上下文"（如泛化的"这首诗表达了什么情感"无法定位作品），按题型将诗题/作者/朝代/正文智能注入问题文本。
- 路径：`data/poetry/qa_pairs/{train,val,test}.json`

## 中华菜谱 QA

- **来源**：大模型两步生成（先生成覆盖八大菜系的菜名清单，再逐菜生成 QA）。
- **规模**：原始约 6000 条，经两个第三方大模型自动清洗（事实性 + 反问质量双维打分）后保留 train 1697 / val 140 / test 160。
- **题型**：食材选择、风味技法、菜系文化、营养提示、家常做法、菜品对比等。
- **字段**：同古诗词，另含 `dish_name` / `cuisine`。
- 路径：`data/recipe/qa_pairs_clean/{train,val,test}.json`

## 法律 QA（民法典）

- **来源**：基于《中华人民共和国民法典》公开文本，法条锚定生成 QA（防幻觉：每条 QA 绑定原文法条）。
- **规模**：train 2409 / val 200 / test 300；另含 MVE 子集 `train_mve.json`(500) / `test_mve.json`(100)。
- **字段**：同上，另含法条引用信息。
- 路径：`data/law/qa_pairs_clean/{train,val,test,train_mve,test_mve}.json`

## 数据生成脚本

- `scripts/gen_qa.py`：古诗词 QA 生成
- `scripts/gen_recipe_v4.py`：菜谱 QA 两步生成
- `scripts/gen_law.py`：民法典 QA 生成（法条锚定）
- `scripts/clean_recipe_data.py`：菜谱裁判自动清洗（支持断点续传/并发）

## 说明

- 菜谱/法律 QA 为大模型生成，可能存在少量事实偏差（已清洗过滤，但不保证 100% 准确）。
- 古诗词原始文本版权归 chinese-poetry 项目及原始出处；民法典为公开法律文本；本仓库仅提供在其上生成的 QA 对。
- MVE/训练强度实验使用各数据集的前 N 条（`--train-limit`），保证可复现。
