"""
后验梯度方向诊断：段间梯度余弦相似度（区分"量级不足" vs "方向冲突"）

用法（已训练模型 + 测试集，无需重跑训练）：
  python -X utf8 grad_conflict_diag.py \
    --model models/recipe_lr5_groupB/groupB_iter5 --group B \
    --test-data data/recipe/qa_pairs_clean/test.json \
    --train-data data/recipe/qa_pairs_clean/train.json \
    --ma-cache cache/recipe_ma_cache.json --limit 100 \
    --out results/grad_conflict_B.txt

判读：
  - cos(g1,g2)/cos(g1,g3)/cos(g2,g3) 平均接近 1 → 段间方向一致，无冲突 → 量级不足是主因
  - 接近 0 或为负 → 方向冲突显著 → PCGrad 式方向消解路线
  - 正但明显 <1（0.3~0.7）→ 部分冲突，量级+方向都需处理
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.training.dialogue import (Group, X_LEVEL, DialogueSample,
                                   build_answer_prompt, build_cq_prompt, build_final_prompt)
from src.training.self_play import build_loss_segments
from src.training.counter_answer import CounterAnswerer, build_pool_from_dataset
from src.counter_question.golden_questions import GoldenQuestions


def load_model(model_path: str):
    import os
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base_name = (_json.load(open(os.path.join(model_path, "adapter_config.json")))
                     .get("base_model_name_or_path") or "Qwen/Qwen2.5-0.5B-Instruct")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        model = PeftModel.from_pretrained(
            base, model_path, torch_dtype=torch.bfloat16).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    # 启用 LoRA 层梯度（诊断需 backward 计算段梯度）
    lora_names = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
    if lora_names:
        for n, p in model.named_parameters():
            p.requires_grad_("lora" in n.lower())
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    return model, tok


def grad_vector(model):
    import torch
    parts = [p.grad.detach().flatten() for p in model.parameters()
             if p.grad is not None and p.requires_grad]
    return torch.cat(parts)


def seg_loss(model, tok, prompt_messages, target_text, max_len=2048):
    import torch
    ids = tok.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = (ids + target_ids)[:max_len]
    labels = ([-100] * len(ids) + target_ids)[:max_len]
    input_ids = torch.tensor([input_ids], device="cuda")
    labels = torch.tensor([labels], device="cuda")
    return model(input_ids=input_ids, labels=labels).loss


def generate(model, tok, messages, max_new=512):
    import torch
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = torch.tensor([list(ids)], device="cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group", choices=["B", "C"], required=True)
    ap.add_argument("--domain", default="recipe")
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--train-data", required=True, help="M_A 检索池")
    ap.add_argument("--ma-cache", default="")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="results/grad_conflict.txt")
    args = ap.parse_args()

    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print(f"[load] {args.model} (group={args.group})", flush=True)
    model, tok = load_model(args.model)

    data = json.load(open(args.test_data, encoding="utf-8"))[:args.limit]
    ca = CounterAnswerer(retrieval_pool=build_pool_from_dataset(args.train_data),
                         llm_client=None, domain=args.domain, cache_path=args.ma_cache or None)
    golden = GoldenQuestions(args.domain) if args.group == "C" else None

    rows = []
    for i, item in enumerate(data):
        q = item["question"]
        ans_first = item.get("answer") or q
        if args.group == "B":
            cq = str(item.get("counter_question", "")).strip() or q
        else:
            cq = golden.pick(item) if golden else q
        r = ca.answer(q, cq)
        ans_final = generate(model, tok, build_final_prompt(args.domain, q, ans_first, cq, r))
        sample = DialogueSample(group=args.group, x_level=X_LEVEL[Group(args.group)],
                                question=q, answer_first=ans_first, counter_question=cq,
                                counter_source="normal" if args.group == "B" else "golden",
                                user_supplement=r, answer_final=ans_final)
        segs = [s for s in build_loss_segments(sample, args.domain) if s["target"]]
        if len(segs) < 2:
            continue
        vecs = []
        for s in segs:
            model.zero_grad()
            loss = seg_loss(model, tok, s["prompt"], s["target"])
            loss.backward()
            vecs.append(grad_vector(model))
        if len(vecs) == 3:
            rows.append((cos(vecs[0], vecs[1]), cos(vecs[0], vecs[2]), cos(vecs[1], vecs[2])))
        elif len(vecs) == 2:
            rows.append((cos(vecs[0], vecs[1]), float("nan"), float("nan")))
        if (i + 1) % 20 == 0:
            print(f"  诊断 {i+1}/{len(data)}", flush=True)

    n = len(rows)
    if n == 0:
        print("无有效样本！")
        return
    c12 = sorted(r[0] for r in rows)
    c13 = sorted(r[1] for r in rows if r[1] == r[1])
    c23 = sorted(r[2] for r in rows if r[2] == r[2])

    def stat(v):
        import statistics
        return (round(statistics.mean(v), 4), round(statistics.median(v), 4),
                round(v[len(v)//10], 4), round(v[len(v)//2], 4), round(v[len(v)*9//10], 4))

    lines = []
    add = lines.append
    add(f"后验梯度方向诊断: {args.model}")
    add(f"样本数={n}  段间余弦（mean/median/p10/p50/p90）")
    if c13 and c23:
        add(f"  cos(初答,反问):   {stat(c12)}")
        add(f"  cos(初答,终答):   {stat(c13)}")
        add(f"  cos(反问,终答):   {stat(c23)}")
    else:
        add(f"  cos(初答,反问):   {stat(c12)}（仅两段）")
    pos = sum(1 for r in rows if r[0] > 0.3)
    neg = sum(1 for r in rows if r[0] < 0)
    add(f"\ncos(初答,反问)>0.3 占比: {pos/n:.1%}   <0 占比: {neg/n:.1%}")
    add("\n判读: 平均余弦>0.7 → 无冲突(量级问题) | 0.3~0.7 → 部分冲突 | <0.3 → 方向冲突为主")
    txt = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
