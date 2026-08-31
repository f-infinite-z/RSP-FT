"""
反问式自博弈训练主循环（v2：A/B/C 三组分层操纵）

RSP-FT 原创核心算法 — 作者独立实现（未经 AI 生成）。
本文实验复现、数据统计与可视化脚本部分借助大模型工具辅助编写初稿，
经人工适配、调试、校验后使用。原创算法框架、实验逻辑、数据校验、
结果分析均由作者独立完成。

对应三组 CE 损失（实验规划 v2 5.2）：
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

class SegmentDiag:
    """梯度诊断记录器：逐 step 记录每段【未加权】损失与梯度范数（LoRA 参数）。
    indep=True 时额外记录 indep_gn（段独立梯度范数，backward 前后梯度快照差），
    供真实 GC（段贡献占比）计算使用——累积范数增量在有方向冲突时失真。
    """

    def __init__(self, path: str, indep: bool = False):
        import csv
        self.f = open(path, "w", encoding="utf-8", newline="")
        self.w = csv.writer(self.f)
        cols = ["iter", "step", "seg", "loss", "gn_cum"]
        if indep:
            cols.append("indep_gn")
        self.w.writerow(cols)
        self.it, self.step = 1, 0
        self.indep = indep

    def set_pos(self, it, step):
        self.it, self.step = it, step

    def write(self, seg_name, loss, gnorm, indep_gn=None):
        row = [self.it, self.step, seg_name, f"{loss:.4f}", f"{gnorm:.4f}"]
        if indep_gn is not None:
            row.append(f"{indep_gn:.4f}")
        self.w.writerow(row)

    def close(self):
        self.f.close()


def cqls_lambda(it: int, iterations: int, mode: str) -> float:
    """CQLS 反问段权重调度：λ(it) 从 0.1 渐升至 1.0。
    none=等权原版；linear=(t)^1；square=(t)^2 慢起步；step=iter<=2 低权后满权。
    """
    if mode == "none":
        return 1.0
    if mode == "step":
        return 0.1 if it <= 2 else 1.0
    t = (it - 1) / max(1, iterations - 1)
    p = 1.0 if mode == "linear" else 2.0
    return 0.1 + 0.9 * (t ** p)


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
        # GradNorm 简化版状态（EMA 每段梯度范数 → 延迟一步权重）
        self._gn_ema = None
        self._gn_weights = None

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

    def _grad_norm(self):
        """当前累积参数梯度范数（仅可训练参数，LoRA 层）。"""
        import torch
        parts = [p.grad.detach().reshape(-1) for p in self.model.parameters()
                 if p.grad is not None and p.requires_grad]
        if not parts:
            return 0.0
        return float(torch.cat(parts).norm().item())

    def _snap_grads(self):
        """backward 前参数梯度快照（用于段独立梯度范数计算）。"""
        import torch
        return {id(p): p.grad.detach().clone()
                for p in self.model.parameters()
                if p.grad is not None and p.requires_grad}

    def _diff_norm(self, snap):
        """段独立梯度范数 = backward 后当前梯度与快照之差（该段刚加入的部分）。"""
        import torch
        parts = []
        for p in self.model.parameters():
            if p.grad is not None and p.requires_grad:
                base = snap.get(id(p))
                g = p.grad.detach()
                d = g if base is None else g - base
                parts.append(d.reshape(-1))
        if not parts:
            return 0.0
        return float(torch.cat(parts).norm().item())

    def train_step(self, optimizer, batch_items, it, scheduler=None, batch_gen=True,
                   diag=None, lam=1.0, final_w=1.0):
        """生成+训练一体（MVE 用）；diag: SegmentDiag 对象；lam: 反问段权重；final_w: 最终回答段权重。"""
        import torch
        self.model.train()
        optimizer.zero_grad()
        total = 0.0
        # 生成阶段：批量构建（提速核心），失败则回退逐条
        if batch_gen:
            samples = self.build_samples_batch(it, batch_items)
        else:
            samples = [self.build_sample(it, item) for item in batch_items]
        for si, sample in enumerate(samples):
            segs = [s for s in build_loss_segments(sample, self.domain) if s["target"]]
            raws = []
            for i, s in enumerate(segs):
                l = self._seg_loss(s["prompt"], s["target"])
                raws.append(float(l.item()))
                if i == 1 and lam != 1.0:
                    l = lam * l
                elif i == 2 and final_w != 1.0:
                    l = final_w * l
                snap = self._snap_grads() if diag is not None and diag.indep else None
                l.backward()
                if diag is not None:
                    indep = self._diff_norm(snap) if snap is not None else None
                    diag.write(s["name"], raws[-1], self._grad_norm(), indep)
            total += sum(raws)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return total / max(1, len(batch_items))

    def _collect_grads(self):
        import torch
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        return params, torch.cat([p.grad.detach().flatten() for p in params])

    def _scatter_grads(self, params, vec):
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = vec[offset:offset + n].view_as(p).clone()
            offset += n

    def loss_step(self, optimizer, samples, scheduler=None, diag=None, lam=1.0, final_w=1.0, pcgrad=False, gradnorm=False):
        """样本已预生成，只算loss+反向+更新（与生成阶段解耦）。

        diag: SegmentDiag 对象——逐段记录【未加权】损失与累积梯度范数（梯度诊断）
        lam: 反问段权重调度系数（lam=1.0 等权=原版行为）
        final_w: 最终回答段权重（>1 抬升被淹没段的绝对梯度贡献，CQLS+）
        pcgrad: PCGrad 方向冲突消解（段间余弦<0 时投影到正交方向，NeurIPS 2020）
        gradnorm: GradNorm 简化版（EMA 段梯度范数倒数自适应权重，延迟一步，ICML 2018 思想）
        """
        import torch
        self.model.train()
        optimizer.zero_grad()
        total = 0.0
        for sample in samples:
            segs = [s for s in build_loss_segments(sample, self.domain) if s["target"]]
            raws = []
            if pcgrad and len(segs) > 1:
                # ---- PCGrad 分支：逐段 backward 取独立梯度 → 两两投影消解 → 累加写回 ----
                params = [p for p in self.model.parameters() if p.requires_grad]
                prev = None
                grads = []
                for i, s in enumerate(segs):
                    l = self._seg_loss(s["prompt"], s["target"])
                    raws.append(float(l.item()))
                    if i == 1 and lam != 1.0:
                        l = lam * l
                    elif i == 2 and final_w != 1.0:
                        l = final_w * l
                    snap = self._snap_grads() if diag is not None and diag.indep else None
                    l.backward()
                    if diag is not None:
                        indep = self._diff_norm(snap) if snap is not None else None
                        diag.write(s["name"], raws[-1], self._grad_norm(), indep)
                    cur = torch.cat([p.grad.detach().flatten() for p in params])
                    grads.append(cur if prev is None else cur - prev)
                    prev = cur
                final = grads[0].clone()
                for i in range(1, len(grads)):
                    gi = grads[i].clone()
                    for j, gj in enumerate(grads):
                        if i == j:
                            continue
                        dot = (gi * gj).sum()
                        if dot < 0:
                            gi = gi - (dot / (gj.norm() ** 2 + 1e-8)) * gj
                    final = final + gi
                self._scatter_grads(params, final)
            elif gradnorm and len(segs) > 1:
                # ---- GradNorm 简化版：上一步 EMA 权重缩放各段 → backward → 收集独立段范数 → 更新 EMA ----
                params = [p for p in self.model.parameters() if p.requires_grad]
                prev = None
                seg_norms = []
                for i, s in enumerate(segs):
                    l = self._seg_loss(s["prompt"], s["target"])
                    raws.append(float(l.item()))
                    if self._gn_weights is not None:
                        l = self._gn_weights[i] * l
                    snap = self._snap_grads() if diag is not None and diag.indep else None
                    l.backward()
                    if diag is not None:
                        indep = self._diff_norm(snap) if snap is not None else None
                        diag.write(s["name"], raws[-1], self._grad_norm(), indep)
                    cur = torch.cat([p.grad.detach().flatten() for p in params])
                    gn = cur if prev is None else (cur - prev)
                    seg_norms.append(float(gn.norm().item()))
                    prev = cur
                alpha = 0.1
                if self._gn_ema is None:
                    self._gn_ema = seg_norms
                else:
                    self._gn_ema = [alpha * n + (1 - alpha) * e
                                    for n, e in zip(seg_norms, self._gn_ema)]
                mean_e = sum(self._gn_ema) / len(self._gn_ema)
                self._gn_weights = [mean_e / (e + 1e-8) for e in self._gn_ema]
            else:
                for i, s in enumerate(segs):
                    l = self._seg_loss(s["prompt"], s["target"])
                    raws.append(float(l.item()))
                    if i == 1 and lam != 1.0:
                        l = lam * l
                    elif i == 2 and final_w != 1.0:
                        l = final_w * l
                    snap = self._snap_grads() if diag is not None and diag.indep else None
                    l.backward()
                    if diag is not None:
                        indep = self._diff_norm(snap) if snap is not None else None
                        diag.write(s["name"], raws[-1], self._grad_norm(), indep)
            total += sum(raws)
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
    支持两种目录：完整模型目录（AutoModel 直接加载）或 PeftModel adapter 目录
    （含 adapter_config.json → 先加载 base 再挂载 LoRA，与 base_sft 的保存格式匹配）。
    加载后显式启用梯度(之前 requires_grad=0 导致 backward 报错)。"""
    import json as _json
    import os
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    adapter_cfg = os.path.join(sft_model, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        base_name = (_json.load(open(adapter_cfg)).get("base_model_name_or_path")
                     or "Qwen/Qwen2.5-0.5B-Instruct")
        print(f"[load] adapter 目录 → base={base_name}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        model = PeftModel.from_pretrained(
            base, sft_model, torch_dtype=torch.bfloat16).to("cuda")
    else:
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
    ap.add_argument("--diag-output", default="",
                    help="梯度诊断CSV路径（逐step记录三段未加权损失+累积梯度范数）")
    ap.add_argument("--schedule", default="none", choices=["none", "linear", "square", "step"],
                    help="CQLS反问段权重调度：none=等权原版 / linear=(t)^1 / square=(t)^2 / step=iter<=2低权")
    ap.add_argument("--final-weight", type=float, default=1.0,
                    help="CQLS+互补权重：最终回答段权重(>1 抬升其梯度贡献，默认1.0=原CQLS)")
    ap.add_argument("--pcgrad", action="store_true",
                    help="PCGrad方向冲突消解：段间余弦<0时投影到正交方向(NeurIPS 2020)")
    ap.add_argument("--gradnorm", action="store_true",
                    help="GradNorm简化版：EMA段梯度范数倒数自适应权重(延迟一步,ICML 2018思想)")
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

    diag = None
    if args.diag_output:
        from pathlib import Path as _P
        _P(args.diag_output).parent.mkdir(parents=True, exist_ok=True)
        diag = SegmentDiag(args.diag_output, indep=True)
        print(f"[diag] 梯度诊断记录 -> {args.diag_output} (schedule={args.schedule}, indep_gn=on)", flush=True)

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
        lam = cqls_lambda(it, args.iterations, args.schedule)
        for i in range(0, len(samples), args.batch_size):
            if diag is not None:
                diag.set_pos(it, i // args.batch_size)
            losses.append(trainer.loss_step(optimizer, samples[i:i + args.batch_size],
                                            scheduler, diag=diag, lam=lam,
                                            final_w=args.final_weight,
                                            pcgrad=args.pcgrad,
                                            gradnorm=args.gradnorm))
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[group {args.group}][iter {it}] avg_loss={sum(losses)/len(losses):.4f} "
              f"lr={cur_lr:.2e} lam={lam:.3f} final_w={args.final_weight:.2f}", flush=True)
        out_dir = Path(args.out) / f"group{args.group}_iter{it}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

    if diag is not None:
        diag.close()
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
