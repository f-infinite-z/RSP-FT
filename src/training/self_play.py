"""
反问式自博弈训练主循环（v2：A/B/C 三组分层操纵）

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

# ============================================================
# 人工批注（作者）
# ============================================================
# 自博弈的主体循环，A组是基座对照的考虑反问的有无影响，B/C这边是控制变量保持3轮一致。C组那边单独做了渐进式训练，目的是为了训练小模型的反问能力，做一个逐级提升
不过后续自由生成阶段很慢，可以适当调整一下比例，不要全部自由生成也许，小模型输出太慢，0.5-3B基本上都慢，我500条5轮C组基本上都要1-2小时
# 这边项目进行的时候用云的3090-24G/A100-40G，生产分逐条生成太慢了，GPU跑不满就10%多，显存到是一半左右，所以做了优化，生成和训练隔离，做预生成，可以节省时间成本，如果
要复现的话，我这边用的48条并行生成吃满GPU，可以根据你的配置直接改适合你的机子的，其他基本上也是实验中踩坑的一些优化，能多吃满配置就吃满吧，时间成本或者云的成本利用率高点好
# ============================================================

对应三组 CE 损失：
  A 组: L = CE(ans | q)
  B 组: L = CE(ans_first | q) + CE(cq_low  | q,ans_first) + CE(ans_final | q,ans_first,cq_low, r)
  C 组: L = CE(ans_first | q) + CE(cq_high | q,ans_first) + CE(ans_final | q,ans_first,cq_high,r)

B / C 损失结构完全一致，唯一差异是反问文本质量（cq_low vs cq_high）。

反问来源：
  B 组 : 数据集已生成的普通反问 counter_question 字段
  C 组 : 黄金三问模板 + 实体填充（渐进式：iter1-2 全模板 / iter3-4 半模板 / iter5 自由生成）

本文件可在无 GPU 环境 import（重依赖延迟导入），便于逻辑审阅与 dry-run。
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from .dialogue import (Group, X_LEVEL, DialogueSample, build_answer_prompt,
                       build_cq_prompt, build_final_prompt)
from .counter_answer import CounterAnswerer, build_pool_from_dataset
from src.counter_question.golden_questions import GoldenQuestions


# ---- C 组渐进式反问来源 -----------------------------------------------------

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


def stage_of_iter(it: int) -> int:
    if it <= 2:
        return 1
    if it <= 4:
        return 2
    return 3


def golden_ratio_for_stage(stage: int) -> float:
    """C 组黄金模板占比：iter1-2=100%，iter3-4=50%，iter5=0%（自由生成）。"""
    return {1: 1.0, 2: 0.5, 3: 0.0}[stage]


# ---- 损失片段规格 -----------------------------------------------------------

def build_loss_segments(sample: DialogueSample, domain: str) -> List[Dict]:
    """
    返回 [{"name", "prompt", "target"}, ...]，与 5.2 三组损失公式一一对应。
    self-play 训练器仅对 target 段做 teacher-forcing CE。
    """
    q = sample.question
    segs = [{
        "name": "ans_first",
        "prompt": build_answer_prompt(domain, q),
        "target": sample.answer_first,
    }]

    if sample.group == Group.A.value:
        return segs

    segs.append({
        "name": "cq",
        "prompt": build_cq_prompt(domain, q, sample.answer_first),
        "target": sample.counter_question or "",
    })
    segs.append({
        "name": "ans_final",
        "prompt": build_final_prompt(
            domain, q, sample.answer_first,
            sample.counter_question or "", sample.user_supplement or ""),
        "target": sample.answer_final or "",
    })
    return segs


# ---- 训练循环 ---------------------------------------------------------------

class SelfPlayTrainer:
    def __init__(self, model, tokenizer, domain, group,
                 counter_answerer: Optional[CounterAnswerer] = None,
                 golden: Optional[GoldenQuestions] = None,
                 max_len=2048, device="cuda"):
        self.model = model
        self.tok = tokenizer
        self.domain = domain
        self.group = Group(group)
        self.ca = counter_answerer
        self.golden = golden or (GoldenQuestions(domain) if group == "C" else None)
        self.max_len = max_len
        self.device = device

    def _generate(self, messages: List[Dict], max_new=512) -> str:
        import torch
        ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        ids = torch.tensor([_to_id_list(ids)], device=self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                      temperature=0.8, top_p=0.95,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def _generate_batch(self, messages_list: List[List[Dict]], max_new=512) -> List[str]:
        """批量生成（左padding），大幅提升 GPU 利用率。空列表返回空。"""
        import torch
        if not messages_list:
            return []
        seqs = [_to_id_list(self.tok.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True)) for m in messages_list]
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        maxlen = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            padn = maxlen - len(s)
            input_ids.append([pad_id] * padn + s)          # 左 padding
            attn.append([0] * padn + [1] * len(s))
        input_ids = torch.tensor(input_ids, device=self.device)
        attn = torch.tensor(attn, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new, do_sample=True,
                temperature=0.8, top_p=0.95, pad_token_id=pad_id)
        gen = out[:, input_ids.shape[1]:]
        return [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    def _make_cq(self, it: int, item: Dict, question: str, answer_first: str) -> str:
        """按组别决定反问来源。"""
        if self.group == Group.B:
            # B 组：数据集普通反问（低质锚点）
            return str(item.get("counter_question", "")).strip() or \
                self._generate(build_cq_prompt(self.domain, question, answer_first), max_new=128)
        # C 组：黄金三问，渐进式
        ratio = golden_ratio_for_stage(stage_of_iter(it))
        if self.golden and random.random() < ratio:
            return self.golden.pick(item)
        return self._generate(build_cq_prompt(self.domain, question, answer_first), max_new=128)

    def build_sample(self, it: int, item: Dict) -> DialogueSample:
        question = item["question"]
        gold = item.get("answer")
        answer_first = gold if gold is not None else \
            self._generate(build_answer_prompt(self.domain, question))

        if self.group == Group.A:
            return DialogueSample(
                group="A", x_level=0, question=question,
                answer_first=answer_first, counter_source="none",
                answer_final=answer_first)

        cq = self._make_cq(it, item, question, answer_first)
        r = self.ca.answer(question, cq) if self.ca else ""
        answer_final = self._generate(build_final_prompt(
            self.domain, question, answer_first, cq, r))
        return DialogueSample(
            group=self.group.value, x_level=X_LEVEL[self.group], question=question,
            answer_first=answer_first, counter_question=cq,
            counter_source=("golden" if self.group == Group.C else "normal"),
            user_supplement=r, answer_final=answer_final)

    def build_samples_batch(self, it: int, items: List[Dict]) -> List[DialogueSample]:
        """批量构建样本：初答用gold、反问查缓存/模板、最终答【批量生成】。
        要求 item 含 gold answer（本项目数据集均有）。A组无需生成。"""
        # 初答：全部用 gold（数据集有 answer 字段）
        firsts = []
        for item in items:
            gold = item.get("answer")
            firsts.append(gold if gold is not None
                          else self._generate(build_answer_prompt(self.domain, item["question"])))

        if self.group == Group.A:
            return [DialogueSample(group="A", x_level=0, question=it_["question"],
                                   answer_first=af, counter_source="none", answer_final=af)
                    for it_, af in zip(items, firsts)]

        # B/C：反问（查缓存/模板，个别需生成的仍逐条——占比小）
        cqs = [self._make_cq(it, item, item["question"], af)
               for item, af in zip(items, firsts)]
        # M_A 补充（查缓存，快）
        rs = [self.ca.answer(item["question"], cq) if self.ca else ""
              for item, cq in zip(items, cqs)]
        # 最终答：批量生成（核心提速点）
        final_prompts = [build_final_prompt(self.domain, item["question"], af, cq, r)
                         for item, af, cq, r in zip(items, firsts, cqs, rs)]
        finals = self._generate_batch(final_prompts)

        return [DialogueSample(
            group=self.group.value, x_level=X_LEVEL[self.group], question=item["question"],
            answer_first=af, counter_question=cq,
            counter_source=("golden" if self.group == Group.C else "normal"),
            user_supplement=r, answer_final=fin)
            for item, af, cq, r, fin in zip(items, firsts, cqs, rs, finals)]

    def _seg_loss(self, prompt_messages, target_text):
        import torch
        prompt_ids = self.tok.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True)
        prompt_ids = _to_id_list(prompt_ids)
        target_ids = self.tok(target_text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        input_ids = (prompt_ids + target_ids)[:self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:self.max_len]
        input_ids = torch.tensor([input_ids], device=self.device)
        labels = torch.tensor([labels], device=self.device)
        return self.model(input_ids=input_ids, labels=labels).loss

    def train_step(self, optimizer, batch_items, it, scheduler=None, batch_gen=True):
        import torch
        self.model.train()
        optimizer.zero_grad()
        total = 0.0
        # 生成阶段：批量构建（提速核心），失败则回退逐条
        if batch_gen:
            samples = self.build_samples_batch(it, batch_items)
        else:
            samples = [self.build_sample(it, item) for item in batch_items]
        for sample in samples:
            segs = build_loss_segments(sample, self.domain)
            loss = sum(self._seg_loss(s["prompt"], s["target"])
                       for s in segs if s["target"])
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total / max(1, len(batch_items))

    def loss_step(self, optimizer, samples, scheduler=None):
        """样本已预生成，只算loss+反向+更新（与生成阶段解耦）。"""
        import torch
        self.model.train()
        optimizer.zero_grad()
        total = 0.0
        for sample in samples:
            segs = build_loss_segments(sample, self.domain)
            loss = sum(self._seg_loss(s["prompt"], s["target"])
                       for s in segs if s["target"])
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total / max(1, len(samples))


def load_qa(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_trainable_model(sft_model: str):
    """加载 SFT 模型并确保参数可训练。
    保持与评测端(run_eval)一致的 AutoModelForCausalLM 加载方式，
    加载后显式启用梯度(之前 requires_grad=0 导致 backward 报错)。"""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        sft_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

    # 启用 LoRA 层梯度(base 权重保持冻结)；若无 LoRA 命名则全量启用
    lora_names = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
    if lora_names:
        for n, p in model.named_parameters():
            p.requires_grad_("lora" in n.lower())
    else:
        for p in model.parameters():
            p.requires_grad_(True)

    model.train()
    n_trainable = sum(p.requires_grad for p in model.parameters())
    print(f"[load] 可训练参数张量数: {n_trainable} (lora层={len(lora_names)})", flush=True)
    if n_trainable == 0:
        raise RuntimeError("没有可训练参数！检查模型加载方式")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--sft-model")
    ap.add_argument("--train-data")
    ap.add_argument("--test-data")
    ap.add_argument("--domain", default="recipe")
    ap.add_argument("--group", default="C", choices=["A", "B", "C"])
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-limit", type=int, default=0,
                    help="限制训练样本量(MVE用,0=全量)。设定后取前N条,保证可复现")
    ap.add_argument("--ma-cache", default="",
                    help="M_A预生成补充库json路径(全量加速,命中则零API等待)")
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                    help="学习率调度：constant(默认,原行为) 或 cosine(余弦退火,应对大lr震荡)")
    ap.add_argument("--warmup-ratio", type=float, default=0.1,
                    help="余弦退火的warmup占比(仅lr-schedule=cosine时生效)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05,
                    help="余弦退火终点lr占峰值的比例(仅cosine生效,如0.05=衰减到峰值5%)")
    ap.add_argument("--no-batch-gen", action="store_true",
                    help="禁用批量生成(回退逐条,调试用)。默认启用批量生成提速")
    ap.add_argument("--gen-batch", type=int, default=48,
                    help="生成阶段的批大小(与训练batch解耦,加大吃满GPU)。默认48")
    ap.add_argument("--out", default="models/selfplay")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)

    if args.dry_run:
        _dry_run(args)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    model = _load_trainable_model(args.sft_model)

    ca = None
    if args.group in ("B", "C"):
        llm_client = None
        if args.api_key:
            from openai import OpenAI
            llm_client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
        pool = build_pool_from_dataset(args.train_data)
        ca = CounterAnswerer(retrieval_pool=pool, llm_client=llm_client, domain=args.domain,
                             cache_path=args.ma_cache or None)

    golden = GoldenQuestions(args.domain) if args.group == "C" else None
    trainer = SelfPlayTrainer(model, tok, args.domain, args.group,
                              counter_answerer=ca, golden=golden)

    data = load_qa(args.train_data)
    if args.train_limit and args.train_limit > 0:
        data = data[:args.train_limit]
        print(f"[train-limit] 训练样本限制为前 {len(data)} 条", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 可选：余弦退火学习率调度（应对大 lr 后期震荡）
    scheduler = None
    if args.lr_schedule == "cosine":
        import math
        steps_per_iter = (len(data) + args.batch_size - 1) // args.batch_size
        total_steps = steps_per_iter * args.iterations
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            # 余弦从 1 衰减到 min_lr_ratio
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[lr-schedule] cosine: total_steps={total_steps}, warmup={warmup_steps}, "
              f"peak_lr={args.lr}, min_lr={args.lr * args.min_lr_ratio:.2e}", flush=True)

    for it in range(1, args.iterations + 1):
        random.shuffle(data)
        # 阶段1：大batch批量生成全部样本（吃满GPU，核心提速）
        samples = []
        gb = args.gen_batch if not args.no_batch_gen else args.batch_size
        for i in range(0, len(data), gb):
            chunk = data[i:i + gb]
            if args.no_batch_gen:
                samples.extend(trainer.build_sample(it, item) for item in chunk)
            else:
                samples.extend(trainer.build_samples_batch(it, chunk))
        # 阶段2：按训练batch算loss并更新
        losses = []
        for i in range(0, len(samples), args.batch_size):
            losses.append(trainer.loss_step(optimizer, samples[i:i + args.batch_size], scheduler))
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[group {args.group}][iter {it}] avg_loss={sum(losses)/len(losses):.4f} lr={cur_lr:.2e}", flush=True)
        out_dir = Path(args.out) / f"group{args.group}_iter{it}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    if ca:
        print("M_A 仿真补充策略统计:", ca.report())


def _dry_run(args):
    """无 GPU：打印三组损失段结构，校验逻辑。"""
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    data = load_qa(args.train_data) if args.train_data else [{
        "question": "赏析《登高》", "answer": "《登高》是杜甫晚年之作……",
        "counter_question": "这首诗押什么韵？",
        "poem_title": "登高", "poem_author": "杜甫", "poem_dynasty": "唐"}]
    item = data[0]
    golden = GoldenQuestions(args.domain)

    for grp in ["A", "B", "C"]:
        if grp == "A":
            cq, src = None, "none"
        elif grp == "B":
            cq, src = item.get("counter_question", ""), "normal"
        else:
            cq, src = golden.pick(item), "golden"
        s = DialogueSample(
            group=grp, x_level=X_LEVEL[Group(grp)], question=item["question"],
            answer_first=item["answer"], counter_question=cq, counter_source=src,
            user_supplement="（仿真用户补充占位）" if grp != "A" else None,
            answer_final="（最终回答占位）" if grp != "A" else item["answer"])
        segs = build_loss_segments(s, args.domain)
        print(f"\n===== 组 {grp} (X={s.x_level}, 反问来源={src}) : {len(segs)} 个损失段 =====")
        if cq:
            print(f"  反问文本: {cq}")
        for seg in segs:
            print(f"  - {seg['name']}: target='{(seg['target'] or '')[:24]}...'")


if __name__ == "__main__":
    main()
