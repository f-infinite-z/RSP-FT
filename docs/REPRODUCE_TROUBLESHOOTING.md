# 复现踩坑实录与排障指南（Reproduction Troubleshooting）

> **背景**：作者于 2026-07-16 在恒源云全新 3090 实例（零环境）上亲自复现全流程。
> 第一次复现（R1）把所有坑踩一遍并如实记录于此；修复后进行第二次复现（R2），要求全流程一次跑通。
> 本文既是复现真实性的过程证据，也为其他复现者提供排障经验。
> **计时口径**：R1 以排障为目的（时间多耗在下载与试错），不作为标准复现耗时；标准耗时以 R2 一次跑通轮为准。

## 复现环境

| 项 | R1 配置 |
|----|---------|
| 云平台 | 恒源云（按量计费） |
| GPU | RTX 3090 24G |
| CPU | AMD EPYC 7601 32-Core |
| 镜像 | 官方镜像 PyTorch 2.4.0 / CUDA 12.1.1 / Python 3.11 |
| 开始时间 | 2026-07-16 19:37 |

---

## 坑清单（R1 实录，按遭遇顺序）

### 坑1：`pip install -r requirements.txt` 报文件不存在

- **现象**：`ERROR: Could not open requirements file: No such file or directory`
- **根因**：零环境云机器上项目还没上传/没进入项目目录，指南早期版本未写明"上传什么、传到哪"
- **修复**：本地打包 `opensource_RSP-FT.tar.gz` → 上传恒源云 `/hy-tmp`（AutoDL 为 `/root/autodl-tmp`）→ 解压 → `cd opensource_RSP-FT` 再执行一切命令
- **状态**：指南已补"二、上传项目"章节 ✅

### 坑2：verify_pipeline.py 崩溃 `total_mem` 属性不存在

- **现象**：`AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
- **根因**：正确属性名是 `total_memory`。此前代码审计在无 GPU 环境进行，这行只在有 GPU 时执行，静态检查无法触达 —— **只有真实 GPU 复现才能暴露**
- **修复**：`sed -i 's/total_mem /total_memory /' verify_pipeline.py`（源码已修）
- **状态**：已修复 ✅

### 坑3：requirements.txt 漏依赖 matplotlib

- **现象**：`import matplotlib` FAIL，连带 `make_pipeline_fig`（出图）FAIL
- **根因**：requirements.txt 未列 matplotlib/openpyxl（原实验环境里恰好装过，依赖清单凭记忆整理有遗漏）
- **修复**：`pip install matplotlib`；requirements.txt 已补 matplotlib/openpyxl
- **状态**：已修复 ✅

### 坑4：`import peft` 报 DTensor 错误（依赖漂移，最典型的坑）

- **现象**：`ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'`
- **误诊过程（保留作教训）**：先以为是 peft 版本太新 → pin `peft<0.14` → 装了 peft==0.13.2 依旧报错
- **真正根因**：完整 traceback 显示错误链是 `peft → transformers 5.14.0 → sharding_utils.py → torch.distributed.tensor.DTensor`。**transformers 5.14.0（复现时 pip 拉到的最新版）要求 torch>=2.5**，而镜像是 torch 2.4.0。`import transformers` 本身 PASS 是因为它懒加载，peft 一触发具体模型模块加载才炸 —— 依赖漂移的隐蔽性所在
- **修复**：升级 torch 至 2.5.1（cu121 wheel，3090 兼容）：
  ```bash
  pip install torch==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
  # 或官方源：pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  ```
- **教训**：① 开源 requirements 只写 `>=` 下限，一年后 pip 拉到的新版组合可能互不兼容，**必须提供 requirements.lock（pip freeze 全量锁定）**；② 报 import 错误先看完整 traceback 链，别只看表层模块名
- **状态**：（待 R1 验证后回填）

### 坑5：SFT 训练崩溃 `pyarrow ArrowTypeError: Expected bytes, got a 'int' object`

- **现象**：verify 第4步 SFT 冒烟 FAIL，traceback 落在 `datasets/arrow_writer.py`
- **根因**：坑4 的续集。transformers 5.14 的 `apply_chat_template(tokenize=True)` 返回 dict（BatchEncoding），`base_sft.py` 用 `list(...)` 包装时拿到的是**键名字符串** `['input_ids','attention_mask']`，与 answer 的 int token 拼接成混合类型列表 → pyarrow 构表报错。项目内 `self_play.py` 早已用 `_to_id_list()` 做了返回类型兼容，但 `base_sft.py` 漏了同款处理——原实验环境的 transformers 返回 list，问题被掩盖
- **修复**：`base_sft.py` 补入与 `self_play.py` 一致的 `_to_id_list()` 兼容函数（源码已修）
- **教训**：同一外部 API 的兼容包装要**全项目统一使用**，只修一处会留下定时炸弹；库的懒加载/版本漂移可能让老代码在新环境里以新的方式失败
- **状态**：已修复 ✅

### 坑6：self_play 加载模型报 `cannot import name '_maybe_shard_state_dict_for_tp'`

- **现象**：transformers 的 peft 集成层（`integrations/peft.py`）import 新版 peft 的私有函数失败
- **根因**：与坑4正好相反——坑4 时误诊 pin 了 `peft==0.13.2`（太老），而 transformers 5.14 的 peft 集成需要**新版 peft**。torch/transformers/peft 三件套必须**成套匹配**，单独 pin 任何一个都会顾此失彼
- **修复**：`pip install -U peft`（torch 已升 2.5.1，新版 peft 的 DTensor 依赖已满足）。最终验证可行的组合：**torch 2.5.1 + transformers 5.14 + peft 最新版**
- **教训**：深度学习三件套（torch/transformers/peft）版本是一个整体，修复不兼容时应对齐"同期发布"的版本组合，而非只动其中一个；requirements.txt 宽松约束不可靠，**以 requirements.lock 全量锁定为准**
- **状态**：已修复 ✅（verify_pipeline 15 PASS / 0 FAIL，剩 1 项评测冒烟需 API key）

### 坑7：`python scripts/pregen_counter_answers.py` 报 `No module named 'src'`

- **现象**：scripts/ 下脚本无法 import 项目内 src 包
- **根因**：脚本内 `sys.path.insert(0, str(Path(__file__).parent))` 插入的是 `scripts/` 自身而非项目根。原实验时这些脚本放在项目根目录运行没问题，开源整理挪入 `scripts/` 子目录后路径失效——**目录重组后没有重新在目标结构下运行验证**
- **修复**：5 个受影响脚本（pregen_counter_answers / t4_gen_grade / t4_gen_grade_locked / gen_qa / clean_recipe_data）统一改为 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
- **教训**：文件重组不是纯搬运，任何路径相关代码都要在新结构下实跑验证
- **状态**：已修复 ✅

### 坑8：`run_eval.py: error: unrecognized arguments: --ma-cache`

- **现象**：复现指南评测命令带 `--ma-cache`，但 `run_eval.py` 不认识该参数
- **根因**：`--ma-cache` 当时只加在了 `self_play.py`（训练端），`run_eval.py`（评测端）漏加，但指南/README 按"两端都支持"来写——**文档与代码能力不同步**
- **修复**：`run_eval.py` 补 `--ma-cache` 参数并传入 `CounterAnswerer(cache_path=...)`（底层 `counter_answer.py` 本就支持缓存，只是入口没暴露）
- **教训**：写使用文档时每条命令都应实际执行一遍，"应该支持"不等于"支持"
- **状态**：已修复 ✅

### 坑9：SFT 7 秒"完成"——开源包里的数据是样例不是完整数据集

- **现象**：完整复现第一步 SFT 仅 3 step / 7 秒结束，train_loss 2.62 明显偏高（正常 ~15 分钟、loss 降至 ~1.5）
- **根因**：开源包 `data/` 下的 json 是**几条样例**（每个 ~5KB），完整数据集（菜谱清洗版 1697 条 ~2.4MB）未入包，复现指南也没写数据获取步骤——流程能"跑通"但跑的不是真实验，**静默无报错是此坑最危险之处**
- **修复**：完整数据集已入包（`data/{poetry/qa_pairs, recipe/qa_pairs_clean, law/qa_pairs_clean}/`），样例移至 `data/examples/` 并明确标注"勿用于训练"；`data/README.md` 重写目录结构说明；指南补数据核对步骤
- **检查方法**：跑 SFT 前先核对数据量：`python -c "import json; print(len(json.load(open('data/recipe/qa_pairs_clean/train.json',encoding='utf-8'))))"` 预期 1697；此外**训练耗时异常短 = 红色警报**，别庆幸"跑得快"
- **教训**：复现验证不仅看"是否报错"，更要核对**规模量级**（数据条数/训练时长/loss水平）与论文声称一致
- **状态**：已修复 ✅（完整数据入包：诗词 4999/500/500、菜谱 1697/140/160、法律 2409/200/300+MVE子集）

### 坑10：B 组训练启动时预生成缓存只载入了一部分（流程编排缺同步点）

- **现象**：`[CounterAnswerer] 载入预生成缓存 870 条`，远少于应有数量——预生成还在后台跑，训练循环已经开始
- **根因**：指南把 pregen 放后台（`nohup ... &`）却**没有等待同步点**，训练 for 循环紧接着启动；且 pregen 未加 `--limit` 与训练 `--train-limit 500` 对齐（在全量上跑，耗时远超预估）；self_play 命令还漏了 `--api-key`，缓存未命中时 LLM 回退不可用，**静默降级**到检索兜底，M_A 生成方式偏离原实验
- **修复**：指南改为 pregen 前台跑完再训练（或 `wait` 同步）；pregen 与训练的数据范围对齐；self_play 补 `--api-key` 兜底。已修
- **教训**：①并行提速的前提是画清依赖边界，"建议后台跑"要写明在哪一步之前必须完成；②带多级回退的系统最怕**静默降级**——跑完必须核对策略统计（`M_A 仿真补充策略统计`中 cache/llm/placeholder 占比）确认走的是预期路径
- **状态**：已修复 ✅

### 坑11：（待续）

---

## 给复现者的建议清单

1. **先跑 `python verify_pipeline.py`**，它按"环境→静态→dry-run→GPU冒烟"分层检查，能在花钱训练前拦住大部分问题
2. **依赖版本以 requirements.lock 为准**（精确复刻作者验证过的组合），requirements.txt 只是宽松下限
3. **恒源云特有**：数据放 `/hy-tmp`（系统盘小）；**关机清数据**，跑完先打包下载结果再关机
4. **国内网络**：`export HF_ENDPOINT=https://hf-mirror.com` 加速模型下载，每个新终端都要重新 export
5. 报 ImportError 先看完整 traceback 最底层，警惕懒加载掩盖的深层依赖不兼容
6. **三裁判配置完先用 curl 各测一发**再进评测（豆包 model 需填方舟的模型/接入点 ID）；评测内置熔断+降级，单裁判故障不会阻塞流程，但主结论应在三裁判齐全下产出

---

## R2 二次复现记录（干净环境一次跑通验证）

**验证范围与计时口径（如实说明）**：
- R2 在全新实例上验证：环境搭建 → verify_pipeline 16 项 → 训练/评测各环节**正常启动运行**（冒烟级）
- 完整训练与评测的结果及耗时以 **R1 修复后的完整运行**为准（三组训练 loss、M_A 策略统计、评测 Y/M 均已对照原实验核验）
- **复现者预期总时长 = R2 环境搭建耗时 + R1 修复后训练评测耗时**（两段拼接，均为实测）

| 项 | R2/R3 配置/结果 |
|----|--------------|
| 开始时间 | R2: 07-17 20:10 / R3: 07-17 21:29（均为全新恒源云 3090 实例，零环境） |
| 环境 | RTX 3090 24G / PyTorch镜像2.4.0 / Python 3.11 |
| 环境搭建耗时 | 依赖安装约5分钟（清华源）+ 基座模型下载数分钟（hf-mirror，需禁xet） |
| verify_pipeline | R2: 环境/静态/dry-run 13项全过；GPU冒烟暴露坑11-13（见下），修复后由 R3 验证 |
| 训练/评测冒烟启动 | R3 逐环节通过：SFT 全量1697条 7m39s loss→1.89 正常收敛；self_play 冒烟通过；评测冒烟 deepseek+qwen 双裁判通过（doubao 404 为方舟接入点 ID 配置问题，属用户配置层面，熔断降级机制正常工作） |
| 与论文一致性（R1数据） | 以 R1 修复后完整运行为准（loss曲线/M_A策略统计/中介分析均已对照原实验核验，见指南批注区） |

**R2/R3 新暴露并已修复的坑**：
- **坑11（安装）**：`requirements.lock` 原含 `+cu121` 本地版本标签，纯 PyPI 源下 `pip install -r` 必失败 → lock 重写为无后缀精简锁定清单，配清华源一条命令可装
- **坑12（依赖残留）**：镜像预装 torchvision/torchaudio 配旧 torch，升级 torch 后版本失配导致 transformers 懒加载崩溃（报错伪装成 `BloomPreTrainedModel` 导入失败）→ 安装序列第一步固定为 `pip uninstall -y torchvision torchaudio`
- **坑13（冒烟超时）**：坑9 完整数据入包后，SFT 冒烟无条数限制跑全量必超 120s → `base_sft.py` 增 `--limit`，verify 与指南 4.1 冒烟均加 `--limit 16`；verify 另增基座模型预下载步骤
- **坑14（下载断流）**：huggingface_hub 新版默认 xet 协议与 hf-mirror 不兼容，下载 1% 即静默中断 → 指南与 verify 均加 `HF_HUB_DISABLE_XET=1`
- **操作提醒**：`HF_ENDPOINT` / `DEEPSEEK_API_KEY` 等环境变量每个新终端都要重新 export；指南占位符 `<key>` 的尖括号不能一起复制
