# 梯度诊断与调度干预（Gradient Diagnosis and Scheduling Intervention）

> 本文档对应论文《小模型多段后训练梯度失衡诊断与调度干预》（改投《计算机应用》版）的方法与复现说明。
> 新增工具：多段后训练梯度失衡诊断工具链 + CQLS 时间调度干预。

## 背景

垂直领域小模型（≤3B）多段后训练（自博弈三段、多任务混合）存在梯度贡献失衡：某段梯度喧宾夺主（占比 >50%）、尾段被相对压制、段间梯度方向冲突（负相关）。外部评测难以定位根因，需要训练动态内部诊断。

## 诊断方法：段独立梯度范数（快照差法）

常用做法是记录累积梯度范数后作增量近似，但 batch 内跨样本累积 + 段间方向抵消会严重低估尾段贡献（实测低估 3~5 倍）。本文采用**梯度快照差分法**：

```
for 每段 s:
    保存当前梯度快照（backward 前）
    段反向传播
    当前梯度 - 快照 = 该段独立梯度分量 → 范数即独立梯度贡献
```

- 训练语义不变：逐段 backward 与 sum-backward 数学等价（梯度线性可加），仅改变测量方式
- 指标：GC（梯度贡献率，段范数/总范数）、段间余弦相似度、三类失衡模式（喧宾夺主 >50%、相对压制 <均值、方向冲突 cos≤0）
- 记录：CSV 逐 step 输出（iter, step, seg, loss, gn_cum, indep_gn）

## 关键结果（0.5B 三段自博弈后训练）

| 组 | iter1 GC（初答:反问:终答） | iter5 GC |
|----|---------------------------|----------|
| B（无调度） | 22:61:17 | 22:56:22 |
| C（无调度） | 22:60:18 | 24:50:26 |
| B+CQLS | **48:15:38** | 20:54:25 |
| C+CQLS | **48:15:37** | 20:57:24 |

- 反问段喧宾夺主（47%~61%）、终答段相对压制（17%~26%）、段间负相关样本占比 41%~64%
- CQLS（反问损失渐进加权调度，λ 0.1→1.0）iter1 将反问段压至 ~15%、终答段抬至 ~38%，是唯一显著有效的干预（C 组 +0.337, p=0.007）
- 普适性：两任务后训练 3:7 失衡、三任务 ~2:5:3（段数加剧尾端压缩）

## 复现步骤

### 1. 三段诊断（自博弈，--diag-output 开关）

```bash
python -m src.training.self_play --config configs/recipe_lr5_groupB.yaml \
    --train-limit 500 --gen-batch 24 --diag-output results/grad_diag/B_diag.csv
python scripts/grad_diag_analyze.py --diag results/grad_diag/B_diag.csv \
    --out results/grad_diag/out_B --seg-names ans_first,cq,ans_final
```

### 2. N 段多任务诊断（普适性验证）

```bash
python scripts/multitask_diag_n.py \
    --sft-model models/recipe_sft \
    --tasks "task_recipe=data/recipe/qa_pairs_clean/train.json=你是中华菜谱与烹饪领域专家。请详细回答下面的问题。;task_poetry=data/poetry/qa_pairs/train.json=你是古诗词鉴赏领域专家。请赏析下面的诗词。" \
    --limit 250 --iterations 5 --lr 5e-6 --batch-size 8 \
    --out models/mt_diag --diag-output results/grad_diag/MT_diag.csv
python scripts/grad_diag_analyze.py --diag results/grad_diag/MT_diag.csv \
    --out results/grad_diag/out_MT --seg-names task_recipe,task_poetry
```

### 3. CQLS 干预（--schedule linear）

```bash
python -m src.training.self_play --config configs/recipe_lr5_groupB_rqls.yaml \
    --train-limit 500 --gen-batch 24 --diag-output results/grad_diag/B_rqls_diag.csv
```

### 4. 方向诊断（段间余弦）

```bash
python scripts/grad_conflict_diag.py --eval results/rqls_B.json --out results/grad_diag/conflict_B.txt
```

## 诊断样例数据

`results/grad_diag_sample/`：
- `B_v2.csv`：B 组（无调度）三段诊断完整轨迹（7500 行，含 indep_gn 列）
- `MT_v2.csv`：两任务诊断轨迹（菜谱+诗词）

## 说明

- 诊断 CSV 需用 `grad_diag_analyze.py` 新版分析（优先读取 indep_gn 列计算真实 GC）；旧版增量近似在方向冲突下失真，仅供参考
- 训练中模型辅助信号（M_A）可用占位符仿真模式（无 API）运行，诊断结论基于组内相对分析
- 实验环境见 `docs/ENVIRONMENT_CLOUD.md`
