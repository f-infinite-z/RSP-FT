# 云环境复现配置（恒源云实测 2026-08-29）

> 适用于在云 GPU 平台（恒源云等）复现 RSP-FT / CQLS 实验。
> 实测通过组合：**Python 3.11 + torch 2.5.1(cu124) + transformers 5.13.0**

## 1. 硬件与镜像

| 项 | 值 |
|----|----|
| 实例 | 恒源云 RTX 3090 Ti-24G（0.8~0.99 元/小时，周末谷期更便宜） |
| 数据盘 | /hy-tmp（关机即清空，**跑完先下载结果**） |
| 基础镜像 | PyTorch 2.4.0 + CUDA 12.1 + Python 3.11 |

## 2. 依赖安装（完整命令，清华源）

```bash
cd /hy-tmp && tar xzf deploy.tar.gz

# 核心组合（实测：必须 torch>=2.5 才能配合 transformers 5.x）
python -m pip install torch==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -q "transformers==5.13.0" openai pyyaml protobuf pandas matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可选（客观指标评测用）
python -m pip install -q rouge-score bert-score jieba scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
# 预期: 2.5.1+cu124 True 5.13.0
```

## 3. 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com        # 国内下载模型走镜像
export HF_HUB_DISABLE_XET=1                     # hf-mirror 时建议关闭 xet 协议
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEEPSEEK_API_KEY=...                     # M_A 补充 + 裁判
export ARK_API_KEY=...                          # 豆包裁判（火山方舟）
export DASHSCOPE_API_KEY=...                    # 千问裁判（阿里百炼）
```

## 4. 启动训练

```bash
nohup bash run_diag_05b.sh > run_diag_05b.log 2>&1 &
tail -f run_diag_05b.log
```

## 5. 踩坑记录（重要）

| 问题 | 现象 | 原因 | 解决 |
|------|------|------|------|
| transformers 被禁用 | `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.0` | 镜像自带 torch 2.4.0，transformers 5.x 要求 ≥2.5 | `pip install torch==2.5.1`（PyPI 版自带 cu124 运行库，无需系统 CUDA） |
| 缺 protobuf | `ImportError: requires the protobuf library` | tokenizer 依赖 | `pip install protobuf` |
| 4.x 旧版不可用 | `AttributeError: 'list' object has no attribute 'keys'`（Qwen2FastTokenizer） | transformers 4.49 与新版 tokenizer_config（extra_special_tokens）不兼容 | **不要用 transformers 4.x**，直接 5.13.0 + torch 2.5.1 |
| pytorch-wheels 镜像无版本 | `Could not find a version that satisfies the requirement torch==2.5.1+cu121` | 清华 pytorch-wheels 镜像缺少该 wheel | 用 PyPI 镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple` 装 `torch==2.5.1` |

## 6. 与其他文档的衔接

- 训练脚本：`run_diag_05b.sh`（诊断 A/B/C → CQLS 调度 B+/C+ → 评测 → 分析 → 打包）
- 训练代码：`src/training/self_play.py`（新增 `--diag-output` 梯度诊断、`--schedule` CQLS 调度）
- 诊断分析：`grad_diag_analyze.py`（三段损失曲线 / 累积梯度范数 / GC 贡献率报告）
- 完整复现步骤见 `REPRODUCE.md`
