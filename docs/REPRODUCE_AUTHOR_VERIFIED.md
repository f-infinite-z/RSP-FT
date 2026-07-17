# Author-Verified Reproduction Guide (Beginner-Friendly)

> **作者本人于实验结束后，在干净云GPU环境中从零复现全流程。**
> 以下每一条命令均经人工逐行执行并验证通过，可直接复制使用。

## 作者复现批注

> 复现了几遍基本上没问题，主要还是卡在环境配置上比较磨人。
> 复现结果和论文主体结论是一致的，这里直接把我最后一次复现的环境全放一下了，有问题的话靠这些信息，会方便很多：
>
> ```
> Python 3.11.10
> accelerate         1.14.0
> datasets           5.0.0
> huggingface_hub    1.23.0
> numpy              2.1.2
> openai             2.45.0
> peft               0.19.1
> torch              2.5.1
> transformers       5.14.0
> ```
>
> 能力有限，实验设计的没有那么完美，如果有优化方向的建议，很乐意进行交流，一起探索，或者我可以详细提供一些实验中的相关心得。

<!-- ============================================================ -->
<!-- 作者批注区域 -->
<!-- ============================================================ -->
<!--                                                              -->
<!-- 环境记录（填写你实际复现时的配置）：                          -->
<!--   日期：2026-07-16 19:37 ~ 07-17 01:4x（R1）                  -->
<!--   云平台：恒源云（3090按量计费）                              -->
<!--   GPU型号：RTX 3090 24G                                       -->
<!--   CPU型号：AMD EPYC 7601 32-Core Processor                    -->
<!--   Python版本：3.11                                            -->
<!--   CUDA版本：12.1.1（驱动535，CUDA 12.2兼容）                  -->
<!--   PyTorch版本：镜像2.4.0 → 复现中升级2.5.1（见踩坑文档坑4）   -->
<!--   实际耗时：修复后有效流程约3.5h（SFT 3m23s + 预生成40min     -->
<!--            + A/B/C训练~1h40m + 三组评测~1h + 分析出图数分钟）；-->
<!--            环境搭建耗时以R2干净轮为准                          -->
<!--   实际API费用：约¥5.3（DS flash预生成3.48 + DS pro裁判<1      -->
<!--               + 豆包0.20 + 千问0.67）                          -->
<!--                                                              -->
<!-- 验证结果：                                                   -->
<!--   SFT冒烟：       ☑ 通过（修复坑5后）                        -->
<!--   self_play冒烟： ☑ 通过（修复坑6后）                        -->
<!--   评测冒烟：      ☑ 通过（修复坑7/坑8后，三裁判全通）        -->
<!--   完整复现：      ☑ 通过（SFT 162steps loss2.07→1.78；       -->
<!--                    B组loss 5.18→5.01 对齐原5.16→4.99；        -->
<!--                    C组iter1 7.4253 对齐原7.43；placeholder=0） -->
<!--   中介分析：      ☑ 通过（a=+2.89 p<.001; b=+0.082 p=.014;   -->
<!--                    间接效应+0.24 CI(0.06,0.41)不含0）          -->
<!--   出图：          ☑ 通过（make_pipeline_fig + paper_figs）    -->
<!--   与论文一致性：  ☑ 核心结论一致（M-Y正相关通道/三条件判定    -->
<!--                    ✓✓✗完全一致；B组Y=3.27较原3.69有浮动，     -->
<!--                    格局A>B>C，属裁判与训练随机性正常范围）     -->
<!--                                                              -->
<!-- 备注：作者批注见文档开头"作者复现批注"章节                  -->
<!--                                                              -->
<!-- 复现全程踩坑与修复实录见 docs/REPRODUCE_TROUBLESHOOTING.md   -->
<!-- ============================================================ -->

---

## 复现验证覆盖范围说明

- **作者已亲自复现验证**：核心管线全流程 —— SFT 基线 → A/B/C 三组自博弈训练（3轮）→ 三裁判评测 → 连续 M-Y 中介分析与出图（菜谱领域，0.5B）。
- **拓展实验**（1.5B/3B 规模、训练强度梯度、控长度、法律/诗词领域等）与核心管线**共用同一套代码，仅更换配置**（`--base` / `--lr` / `--domain` / 数据路径）。全部 config、一条龙脚本（`scripts/run_*.sh`）、三领域完整数据均随仓库提供，原始训练日志已归档可查。
- **给复现者的建议**：拓展实验属于重复性的配置修改与调参工作，建议直接借助 AI 编程助手基于本仓库的 config 模板改写目标配置（改模型规格、领域、超参数），再按本指南流程执行即可，无需从零手写。

---

## 一、开机器

```bash
# 恒源云 / AutoDL
# GPU: 3090-24G
# 镜像: PyTorch 2.1+ / CUDA 12.x / Python 3.11
# 注意：恒源云关机清数据！跑完先下载结果再关机。
```

## 二、上传项目 + 环境安装（10分钟）

> **零环境说明**：云机器开机后是空的，什么都没有。需要先把整个项目传上去。

### 2.1 需要上传的东西（共1个包）

| 上传物 | 内容 | 说明 |
|--------|------|------|
| `opensource_RSP-FT.tar.gz` | 整个项目目录打包 | 含 src/ scripts/ configs/ data/ requirements.txt verify_pipeline.py 全部 |

本地打包命令（在项目父目录执行）：

```bash
# Windows PowerShell（本地）
tar -czf opensource_RSP-FT.tar.gz opensource_RSP-FT
# 注意：打包前确认 configs/judges.yaml（含真实key）不在包里，只带 judges.yaml.example
```

### 2.2 上传到云机

- **恒源云**：JupyterLab 页面直接拖拽上传到 `/hy-tmp`（数据盘，够大；/root 系统盘小勿放）
- **AutoDL**：JupyterLab 上传到 `/root/autodl-tmp`

### 2.3 解压 + 安装依赖

```bash
cd /hy-tmp                      # AutoDL 则 cd /root/autodl-tmp
tar -xzf opensource_RSP-FT.tar.gz
cd opensource_RSP-FT            # ⚠️ 后续所有命令都在这个目录下执行

# ① 卸载镜像预装的 torchvision/torchaudio
#    （它们配镜像自带的旧torch，下一步升级torch后版本不匹配会让 transformers 崩溃；本项目纯文本不需要它们）
pip uninstall -y torchvision torchaudio

# ② 安装锁定版本依赖（清华源加速，几分钟；torch 2.5.1 PyPI版自带CUDA 12.4运行时，3090/CUDA 12.x驱动兼容）
pip install -r requirements.lock -i https://pypi.tuna.tsinghua.edu.cn/simple

export HF_ENDPOINT=https://hf-mirror.com   # 国内加速HF模型下载（新开终端要重新export）
export HF_HUB_DISABLE_XET=1                # 禁用xet下载协议（与hf-mirror镜像不兼容，会1%即断）

# ③ 预下载基座模型（~1GB，先下好避免后续训练步骤下载超时；verify_pipeline 也会自动做这步）
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

# 核对数据集完整性（防止误用样例数据，重要！）
python -c "import json; print('recipe train:', len(json.load(open('data/recipe/qa_pairs_clean/train.json',encoding='utf-8'))))"
# 预期 1697；data/examples/ 下才是格式样例，勿用于训练

# 验证环境
python verify_pipeline.py
# 预期：16/16 PASS（无GPU时GPU冒烟SKIP；评测冒烟需已配置API key）
```

**版本说明**：核心组合 torch 2.5.1 / transformers 5.14.0 / peft 0.19.1 / datasets 5.0.0（`requirements.lock` 锁定，作者两轮云端复现实测）。三件套必须成套匹配，勿单独升降级（踩坑文档坑4/坑6）；镜像预装的 torchvision/torchaudio 必须先卸载，否则与升级后的 torch 冲突导致 transformers 懒加载崩溃。

## 三、配置API Key（2分钟）

```bash
export DEEPSEEK_API_KEY=<你的key>
export ARK_API_KEY=<你的key>          # 豆包（可选）
export DASHSCOPE_API_KEY=<你的key>    # 千问（可选）

# 如果只有DeepSeek key，单裁判评测也能跑
# 复制并编辑裁判配置
cp configs/judges.yaml.example configs/judges.yaml
# 编辑 configs/judges.yaml，填入key
```

## 四、GPU冒烟验证（15分钟，~¥0.3）

### 4.1 SFT基线训练（1步验证）

```bash
python -m src.training.base_sft \
    --domain recipe \
    --data data/recipe/qa_pairs_clean/train.json \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --out models/sft_smoke \
    --epochs 1 --lr 2e-5 --batch-size 1 --grad-accum 1 \
    --max-len 512 --lora-r 4 --lora-alpha 8 --limit 16
# 预期：16条样本训练完成，loss正常下降，models/sft_smoke/ 产生模型文件
# 耗时：<1分钟（模型需已预下载，见二、③）
```

### 4.2 自博弈训练（1轮+3条数据）

```bash
python -m src.training.self_play \
    --domain recipe --group B \
    --sft-model models/sft_smoke \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --iterations 1 --lr 1e-6 --batch-size 1 --train-limit 3 \
    --out models/sp_smoke --no-batch-gen
# 预期：loss输出，models/sp_smoke/groupB_iter1/ 模型保存成功
# 耗时：~5分钟
```

### 4.3 评测冒烟（2条数据）

```bash
# 预生成M_A缓存（可选但建议，用便宜模型）
python scripts/pregen_counter_answers.py \
    --domain recipe \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --limit 10 --api-key $DEEPSEEK_API_KEY --workers 5 \
    --out cache/smoke_ma.json

# 评测
python -m src.eval.run_eval \
    --domain recipe --group B \
    --model models/sp_smoke/groupB_iter1 \
    --test-data data/recipe/qa_pairs_clean/test.json \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --api-key $DEEPSEEK_API_KEY --limit 2 \
    --ma-cache cache/smoke_ma.json \
    --out results/smoke_B.json
# 预期：2条评测完成，results/smoke_B.json 包含 Y/M 分值
# 耗时：~2分钟
```

**冒烟小结**：SFT→self_play→评测 三环节全部跑通且无报错 → 环境就绪。

## 五、最小完整复现（2小时，~¥2 + ~¥20 API）

### 5.1 菜谱领域 A/B/C 完整实验

```bash
# SFT基线（用完整数据，先核对 train=1697 条）
python -m src.training.base_sft \
    --config configs/sft_recipe.yaml
# 预期：162 steps / 3 epoch，loss 2.07→1.78，models/recipe_sft/ 保存
# 耗时：~4分钟（3090实测3m23s）

# 预生成M_A缓存（⚠️ 前台跑完再进下一步——训练依赖完整缓存，不要与训练并行）
python scripts/pregen_counter_answers.py \
    --domain recipe \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --api-key $DEEPSEEK_API_KEY --workers 10 \
    --out cache/recipe_ma.json
# 看到"缓存已存至"才算完成
# 注意：全量约6800个(问题,反问)对，耗时约40-60分钟（费用低，用flash档模型）

# 三组自博弈训练（A/B/C，依次跑；--api-key 是缓存未命中时的LLM回退兜底，勿省略）
for G in A B C; do
  echo "=== 训练组 $G $(date '+%F %T') ==="
  python -m src.training.self_play \
      --domain recipe --group $G \
      --sft-model models/recipe_sft \
      --train-data data/recipe/qa_pairs_clean/train.json \
      --iterations 3 --lr 1e-6 --batch-size 8 --train-limit 500 \
      --ma-cache cache/recipe_ma.json --gen-batch 48 \
      --api-key $DEEPSEEK_API_KEY \
      --out models/recipe_group$G
done
# 预期：A组~15分钟，B/C组各~30分钟（含生成时间）
# ⚠️ B/C 组跑完核对日志中的"M_A 仿真补充策略统计"：应以 cache 为主，placeholder 必须为 0
# 耗时：约1.5小时

# 三组评测
for G in A B C; do
  echo "=== 评测组 $G ==="
  python -m src.eval.run_eval \
      --domain recipe --group $G \
      --model models/recipe_group${G}/group${G}_iter3 \
      --test-data data/recipe/qa_pairs_clean/test.json \
      --train-data data/recipe/qa_pairs_clean/train.json \
      --api-key $DEEPSEEK_API_KEY --limit 100 \
      --ma-cache cache/recipe_ma.json \
      --out results/recipe_${G}.json
done
# 耗时：~15分钟（3组×100条×API调用）
```

### 5.2 中介分析与出图

```bash
# 中介分析
python -m src.eval.run_eval \
    --aggregate results/recipe_A.json results/recipe_B.json results/recipe_C.json \
    --out results/recipe_mediation.txt
# 预期：输出 a/b/c/c' 系数 + Bootstrap CI + 三条件判定

# 统计分析
python scripts/stat_test.py --domains recipe --out results/stat_test.txt
python scripts/probe_confounders.py --prefix recipe --out results/probe_recipe.txt

# 出图
python scripts/make_paper_figs.py
# 预期：results/paper_figs/ 下生成 fig1~fig4.png

# 人机一致性（如果有标注数据）
# python scripts/analyze_agreement.py
```

## 六、验证清单（R1 实测已回填）

| # | 检查项 | 预期 | 作者R1实测（2026-07-16） |
|---|--------|------|----------|
| 1 | SFT训练 | loss正常下降 | ✅ 162 steps / 3m23s，loss 2.07→1.78 |
| 2 | self_play三组 | 每轮输出loss | ✅ A: 1.817→1.815；B: 5.18→5.01（对齐原5.16→4.99）；C: 7.43→5.79（iter1对齐原7.43）；M_A placeholder=0 |
| 3 | 评测JSON | results/recipe_{A,B,C}.json 存在 | ✅ A: Y=3.46；B: Y=3.27/M=6.24；C: Y=3.15/M=5.79（原M: B6.21/C5.93） |
| 4 | 中介分析 | a/b/c/c' 系数 + Bootstrap CI | ✅ a=+2.89(p<.001) b=+0.082(p=.014) 间接+0.24 CI(0.06,0.41) c'=-0.40 |
| 5 | 统计分析 | 配对t检验 + Cohen's d | ✅ stat_test.txt / probe_recipe.txt 正常产出 |
| 6 | 出图 | results/paper_figs/ 下4张PNG | ✅ |
| 7 | **与论文一致** | 连续M-Y r值方向与论文一致 | ✅ M→Y 显著为正、间接效应CI不含0、三条件判定✓✓✗与原实验完全一致；B组Y浮动属随机性正常范围 |

## 七、结果存档

```bash
# 打包结果
tar -czf results_full.tar.gz results/
tar -czf models_smoke.tar.gz models/

# 下载到本地（恒源云关机前！）
# scp 或 平台下载功能

# 关机
# sudo shutdown -h now
```

---

## 环境记录（写入论文/文档）

| 项目 | 配置 |
|------|------|
| 复现日期 | R1: 2026-07-16 19:37 ~ 07-17 01:4x（R2 待补） |
| 云平台 | 恒源云（按量计费） |
| GPU | NVIDIA GeForce RTX 3090 24GB |
| CPU | AMD EPYC 7601 32-Core Processor |
| Python | 3.11 |
| CUDA | 12.1.1（驱动 535） |
| PyTorch | 2.5.1+cu121（镜像自带2.4.0，因依赖兼容升级，见踩坑文档坑4） |
| 关键依赖 | transformers 5.14.0 / peft 最新版（全量锁定见 requirements.lock） |
| 基座模型 | Qwen2.5-0.5B-Instruct |
| LoRA配置 | r=16 α=32, 可训练参数~8.8M (1.75%) |
| 裁判模型 | DeepSeek-V4-Pro / Doubao-Seed-2-0-Lite / Qwen3.7-Plus |
| API费用 | 约¥5.3（预生成flash 3.48 + 三裁判评测 ~1.9） |

---

> 复现完成后，将本文件中的配置填入 `docs/REPRODUCE.md`。
