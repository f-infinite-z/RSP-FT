"""
非反问多任务梯度诊断（通用 N 段版）：0.5B LoRA 多任务 CE 联合训练
验证"多段后训练梯度失衡"是否与段数/任务目标对齐度相关（隔离变量实验）

用法：
  python -X utf8 multitask_diag_n.py \
    --sft-model models/recipe_sft \
    --tasks "task_recipe=data/recipe/qa_pairs_clean/train.json=你是中华菜谱与烹饪领域专家。请详细回答下面的问题。;task_poetry=data/poetry/qa_pairs/train.json=你是古诗词鉴赏领域专家。请赏析下面的诗词。;task_law=data/law/qa_pairs_clean/train.json=你是法律领域专家。请依据民法典知识回答下面的问题。" \
    --limit 250 --iterations 5 --batch-size 8 \
    --out models/mt3_diag --diag-output results/grad_diag/MT3_diag.csv

输出：诊断 CSV（每任务一段：task_recipe / task_poetry / task_law ...）+ 每 iter avg_loss
"""
import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.training.self_play import SegmentDiag


def load_qa(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seg_loss(model, tok, messages, target_text, max_len=2048):
    import torch
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = (ids + target_ids)[:max_len]
    labels = ([-100] * len(ids) + target_ids)[:max_len]
    input_ids = torch.tensor([input_ids], device="cuda")
    labels = torch.tensor([labels], device="cuda")
    return model(input_ids=input_ids, labels=labels).loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", required=True)
    ap.add_argument("--tasks", required=True,
                    help="分号分隔任务，每任务 name=path=sys_prompt，如 "
                         "t1=data/a.json=你是xx专家。;t2=data/b.json=你是yy专家。")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="models/mt3_diag")
    ap.add_argument("--diag-output", default="")
    args = ap.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    adapter_cfg = Path(args.sft_model) / "adapter_config.json"
    if adapter_cfg.exists():
        base_name = (json.load(open(adapter_cfg)).get("base_model_name_or_path")
                     or "Qwen/Qwen2.5-0.5B-Instruct")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        model = PeftModel.from_pretrained(base, args.sft_model, torch_dtype=torch.bfloat16).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.sft_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    lora_names = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n.lower())
    model.train()
    print(f"[load] 可训练参数: {sum(p.requires_grad for p in model.parameters())} (lora={len(lora_names)})", flush=True)

    tasks = []
    for spec in args.tasks.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, path, sys_prompt = spec.split("=", 2)
        data = load_qa(path)[:args.limit]
        tasks.append({"name": name, "sys": sys_prompt, "data": data})
        print(f"[data] {name}: {len(data)} 条 <- {path}", flush=True)

    n_tasks = len(tasks)
    if n_tasks < 2:
        print("[ERROR] 至少需要 2 个任务", flush=True)
        sys.exit(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    diag = None
    if args.diag_output:
        Path(args.diag_output).parent.mkdir(parents=True, exist_ok=True)
        diag = SegmentDiag(args.diag_output, indep=True)

    def grad_norm():
        parts = [p.grad.detach().reshape(-1) for p in model.parameters()
                 if p.grad is not None and p.requires_grad]
        return float(torch.cat(parts).norm().item()) if parts else 0.0

    def snap_grads():
        return {id(p): p.grad.detach().clone()
                for p in model.parameters()
                if p.grad is not None and p.requires_grad}

    def diff_norm(snap):
        parts = []
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                base = snap.get(id(p))
                g = p.grad.detach()
                d = g if base is None else g - base
                parts.append(d.reshape(-1))
        return float(torch.cat(parts).norm().item()) if parts else 0.0

    for it in range(1, args.iterations + 1):
        for t in tasks:
            random.shuffle(t["data"])
        samples = []
        for i in range(max(len(t["data"]) for t in tasks)):
            row = []
            for t in tasks:
                d = t["data"][i % len(t["data"])]
                row.append({"name": t["name"], "sys": t["sys"], "item": d})
            samples.append(row)
        total = 0.0
        for i in range(0, len(samples), args.batch_size):
            chunk = samples[i:i + args.batch_size]
            optimizer.zero_grad()
            for srow in chunk:
                # 逐段 backward（与 self_play 一致）：每段记录独立范数（快照差）+ 累积范数
                for s in srow:
                    q = s["item"].get("question") or s["item"].get("prompt")
                    a = s["item"].get("answer") or s["item"].get("target")
                    messages = [{"role": "system", "content": s["sys"]},
                                {"role": "user", "content": q}]
                    l = seg_loss(model, tok, messages, a)
                    snap = snap_grads()
                    l.backward()
                    if diag is not None:
                        diag.set_pos(it, i // args.batch_size)
                        diag.write(s["name"], float(l.item()), grad_norm(), diff_norm(snap))
                    total += float(l.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"[MT][iter {it}] avg_loss={total / (n_tasks * len(samples)):.4f}", flush=True)
        out_dir = Path(args.out) / f"iter{it}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    if diag is not None:
        diag.close()
    print("[MT] 完成")


if __name__ == "__main__":
    main()
