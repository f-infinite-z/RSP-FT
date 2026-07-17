"""
SFT 基线训练（M_0 起点模型）

本质：用 HuggingFace transformers + peft 的标准 LoRA 微调流程，
把 Qwen2.5 基座在各领域训练集上做监督微调，产出 M_0。

全部使用 HuggingFace 官方 API（Trainer / LoraConfig / DataCollatorForSeq2Seq），
无自定义训练逻辑。M_0 是所有 self_play 实验的唯一起点——三组从同一 SFT 模型出发，
保证组间公平可比。

# ============================================================
# 人工批注（作者）
# ============================================================
# 固定范式没啥想说的，验过没问题，注意环境就行
# ============================================================

用法：
  python -m src.training.base_sft --config configs/sft_recipe.yaml
  python -m src.training.base_sft --data data/recipe/qa_pairs/train.json \
      --base Qwen/Qwen2.5-0.5B-Instruct --out models/recipe_sft --domain recipe

依赖：torch transformers peft datasets accelerate（Phase 2 在 GPU 机器安装）
"""
import argparse
import json
import os
from pathlib import Path

DOMAIN_NAME = {"recipe": "中华菜谱与烹饪", "poetry": "古诗词鉴赏"}


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_id_list(x):
    """把 apply_chat_template 的返回统一成 python int list。
    兼容 list / BatchEncoding / dict / torch.Tensor / numpy 等返回类型。"""
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    while isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
        x = x[0]
    return list(x)


def build_sft_examples(data_path, domain, tokenizer, max_len, limit=0):
    """把 QA 对转成 chat 格式的 SFT 样本（仅对 answer 段计算损失）。"""
    from datasets import Dataset

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if limit and limit > 0:
        raw = raw[:limit]
        print(f"[limit] SFT 训练样本限制为前 {len(raw)} 条（冒烟/调试）", flush=True)

    sys_msg = f"你是{DOMAIN_NAME.get(domain, domain)}领域的专家，请准确、完整、有深度地回答用户的问题。"

    input_ids_list, labels_list = [], []
    for item in raw:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q or not a:
            continue
        prompt_msgs = [{"role": "system", "content": sys_msg},
                       {"role": "user", "content": q}]
        prompt_ids = _to_id_list(tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True))
        answer_ids = tokenizer(a, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--data")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="models/sft")
    ap.add_argument("--domain", default="recipe")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="限制训练样本条数（0=全量；仅用于冒烟/调试）")
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer)
    from transformers import DataCollatorForSeq2Seq

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    ds = build_sft_examples(args.data, args.domain, tokenizer, args.max_len, args.limit)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"SFT 完成，模型保存至 {args.out}")


if __name__ == "__main__":
    main()
