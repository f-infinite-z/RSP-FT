"""
评测执行脚本

核心算法逻辑由作者通过自然语言推导完成，AI 工具辅助工程化落地。
所有代码经作者人工逐行校验通过。

# ============================================================
# 人工批注（作者）
# ============================================================
# 这边就是包装流程，整个裁判阶段的，接入-生成对话-裁判打分-返回-中介分析完整流程
# 加载模型 → 按组生成对话 → 三裁判打分 → 落盘JSON → 中介分析
# 两个运行模式：--dry-run（无GPU/API验证流水线）
# 云环境可以先无卡跑一下，省点开销，留了开关
# ============================================================

流水线：
  1. 载入某组已训练模型（或 SFT 基线），在测试集上按 A/B/C 组流程生成对话
  2. 三裁判对「最终回答」打回答质量 Y，对「反问」打反问质量 M
  3. 汇总为中介分析所需的 {group, X, M, Y} 记录，落盘
  4. 三组记录合并后调用 mediation 输出链式中介报告

支持两种运行模式：
  --dry-run     : 不加载模型/不调 API，用模拟评分验证整条流水线与中介分析
  正常模式       : 需 GPU 加载模型 + 裁判 API key

裁判可复用同一 DeepSeek key（DeepSeek 单裁判），或配置多裁判 key 做三裁判交叉。

用法：
  python -m src.eval.run_eval --dry-run --domain recipe
  python -m src.eval.run_eval --domain recipe --group C \
      --model models/recipe_groupC/groupC_iter5 \
      --test-data data/recipe/qa_pairs/test.json \
      --judge-key sk-xxx --out results/recipe_C.json
  python -m src.eval.run_eval --aggregate results/recipe_A.json results/recipe_B.json results/recipe_C.json \
      --out results/recipe_mediation.txt
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.training.dialogue import (Group, X_LEVEL, DialogueSample,
                                   build_answer_prompt, build_cq_prompt, build_final_prompt)
from src.training.counter_answer import CounterAnswerer, build_pool_from_dataset
from src.counter_question.golden_questions import GoldenQuestions
from src.judge.llm_judge import Judge, cross_judge, expertise_aggregate, kendall_w
from src.analysis.mediation import run_mediation


# ---- 记录结构 ---------------------------------------------------------------

def make_record(sample: DialogueSample, y_result: Dict, m_result: Optional[Dict]) -> Dict:
    """组织为落盘记录，含中介分析所需 group/X/M/Y + 明细。"""
    return {
        "group": sample.group,
        "x_level": sample.x_level,
        "question": sample.question,
        "counter_question": sample.counter_question,
        "eval_answer": sample.eval_answer,
        "Y": y_result.get("mean_overall"),
        "M": (m_result.get("mean_overall") if m_result else 0),
        "Y_detail": y_result,
        "M_detail": m_result,
    }


# ---- 生成 + 评分 ------------------------------------------------------------

class Evaluator:
    def __init__(self, model, tokenizer, domain, group, judges: List[Judge],
                 counter_answerer=None, golden=None, device="cuda", max_new=512):
        self.model = model
        self.tok = tokenizer
        self.domain = domain
        self.group = Group(group)
        self.judges = judges
        self.ca = counter_answerer
        self.golden = golden
        self.device = device
        self.max_new = max_new

    def _generate(self, messages, max_new=None):
        import torch
        ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        if not (hasattr(ids, "shape") and hasattr(ids, "to")):
            ids = self.tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=False)
            if not isinstance(ids, list):
                ids = list(ids)
            while isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = torch.tensor([ids], device=self.device)
        else:
            ids = ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new or self.max_new,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def gen_dialogue(self, item) -> DialogueSample:
        q = item["question"]
        answer_first = self._generate(build_answer_prompt(self.domain, q))
        if self.group == Group.A:
            return DialogueSample(group="A", x_level=0, question=q,
                                  answer_first=answer_first, counter_source="none",
                                  answer_final=answer_first)
        if self.group == Group.B:
            cq = str(item.get("counter_question", "")).strip() or \
                self._generate(build_cq_prompt(self.domain, q, answer_first), 128)
            src = "normal"
        else:
            cq = self.golden.pick(item) if self.golden else \
                self._generate(build_cq_prompt(self.domain, q, answer_first), 128)
            src = "golden"
        r = self.ca.answer(q, cq) if self.ca else ""
        answer_final = self._generate(build_final_prompt(self.domain, q, answer_first, cq, r))
        return DialogueSample(group=self.group.value, x_level=X_LEVEL[self.group],
                              question=q, answer_first=answer_first, counter_question=cq,
                              counter_source=src, user_supplement=r, answer_final=answer_final)

    def score(self, sample: DialogueSample) -> Dict:
        y_scores = {}
        m_scores = {}
        for j in self.judges:
            name = j.name or j.model
            y_scores[name] = j.score_answer(sample.question, sample.eval_answer)
            if sample.counter_question:
                m_scores[name] = j.score_cq(sample.question, sample.counter_question)
        y_res = cross_judge(y_scores)
        m_res = cross_judge(m_scores) if m_scores else None
        return make_record(sample, y_res, m_res)

    def run(self, test_data, limit=None, savepoint_path: Optional[str] = None) -> List[Dict]:
        data = test_data[:limit] if limit else test_data
        records = []
        start = 0

        # 断点续传：检查已有 savepoint
        if savepoint_path and Path(savepoint_path).exists():
            try:
                saved = json.load(open(savepoint_path, encoding="utf-8"))
                records = saved.get("records", [])
                start = saved.get("progress", 0)
                if start < len(data):
                    print(f"  [savepoint] 从断点恢复，已完成 {start}/{len(data)}")
            except Exception:
                pass

        try:
            for i, item in enumerate(data[start:], start=start):
                sample = self.gen_dialogue(item)
                records.append(self.score(sample))
                if (i + 1) % 20 == 0:
                    print(f"  评测 {i+1}/{len(data)}", flush=True)
                # 每 50 条保存断点
                if savepoint_path and (i + 1) % 50 == 0:
                    _save_savepoint(savepoint_path, records, i + 1)
        except KeyboardInterrupt:
            print(f"\n  [savepoint] 手动中断，已保存进度 {len(records)}/{len(data)}")
            if savepoint_path:
                _save_savepoint(savepoint_path, records, len(records))
            raise
        except Exception as e:
            print(f"\n  [savepoint] 异常中断({e})，已保存进度 {len(records)}/{len(data)}")
            if savepoint_path:
                _save_savepoint(savepoint_path, records, len(records))
            raise

        # 完成后清理 savepoint
        if savepoint_path and Path(savepoint_path).exists():
            Path(savepoint_path).unlink()
        return records


def _save_savepoint(path: str, records: list, progress: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    json.dump({"records": records, "progress": progress}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    Path(tmp).replace(path)  # 原子替换，避免写入中断导致文件损坏


# ---- 对比+CoT 评测（一次调用比较 A/B/C 三组）----

def compare_evaluate(files_a: str, files_b: str, files_c: str, judges: List[Judge],
                     out_path: Optional[str] = None, limit: int = 0):
    """
    加载三组独立评测结果，按 question 对齐后调用 compare_answer/compare_cq 做对比+CoT 评分。
    files_a/b/c: 可以是单文件或逗号分隔的多文件列表。
    """
    def _load_multi(paths: str) -> List[Dict]:
        recs = []
        for p in paths.split(","):
            recs.extend(json.load(open(p.strip(), encoding="utf-8")))
        return recs

    recs_a = _load_multi(files_a)
    recs_b = _load_multi(files_b)
    recs_c = _load_multi(files_c)

    # 按 question 对齐
    idx_a = {r["question"]: r for r in recs_a}
    idx_b = {r["question"]: r for r in recs_b}
    idx_c = {r["question"]: r for r in recs_c}
    common = sorted(set(idx_a) & set(idx_b) & set(idx_c))
    if limit > 0:
        common = common[:limit]

    print(f"对比+CoT 评测：{len(common)} 条共有问题, {len(judges)} 裁判")
    records = []
    for i, q in enumerate(common):
        ra, rb, rc = idx_a[q], idx_b[q], idx_c[q]
        # 对比回答 Y
        y_result = {"per_judge": {}, "disagreements": 0, "n_judges": len(judges)}
        y_vals = {"A": [], "B": [], "C": []}
        for j in judges:
            name = j.name or j.model
            cmp = j.compare_answer(q, ra["eval_answer"], rb["eval_answer"], rc["eval_answer"])
            if cmp and all(k in cmp for k in ("A", "B", "C")):
                y_result["per_judge"][name] = {k: v.to_dict() for k, v in cmp.items()}
                for k in ("A", "B", "C"):
                    y_vals[k].append(cmp[k].overall)
            else:
                y_result["per_judge"][name] = None
        # 检查分岐
        if len(judges) >= 2:
            diffs = []
            for k in ("A", "B", "C"):
                if len(y_vals[k]) >= 2:
                    diffs.append(max(y_vals[k]) - min(y_vals[k]))
            y_result["disagreements"] = sum(1 for d in diffs if d > 2.0)
            y_result["max_diff"] = max(diffs) if diffs else 0.0
        for k in ("A", "B", "C"):
            y_result[f"mean_{k}"] = round(sum(y_vals[k])/len(y_vals[k]), 3) if y_vals[k] else None

        # 对比反问 M（只比较 B/C）
        m_result = {"per_judge": {}, "disagreements": 0}
        m_vals = {"B": [], "C": []}
        cq_b = rb.get("counter_question", "")
        cq_c = rc.get("counter_question", "")
        if cq_b and cq_c:
            for j in judges:
                name = j.name or j.model
                cmp = j.compare_cq(q, cq_b, cq_c)
                if cmp and all(k in cmp for k in ("B", "C")):
                    m_result["per_judge"][name] = {k: v.to_dict() for k, v in cmp.items()}
                    for k in ("B", "C"):
                        m_vals[k].append(cmp[k].overall)
                else:
                    m_result["per_judge"][name] = None
            if len(judges) >= 2 and all(len(m_vals[k]) >= 2 for k in ("B", "C")):
                m_diffs = [max(m_vals[k]) - min(m_vals[k]) for k in ("B", "C")]
                m_result["disagreements"] = sum(1 for d in m_diffs if d > 2.0)
                m_result["max_diff"] = max(m_diffs)
            for k in ("B", "C"):
                m_result[f"mean_{k}"] = round(sum(m_vals[k])/len(m_vals[k]), 3) if m_vals[k] else None

        records.append({
            "question": q,
            "group_A": ra["group"], "group_B": rb["group"], "group_C": rc["group"],
            "Y_A": y_result.get("mean_A"), "Y_B": y_result.get("mean_B"), "Y_C": y_result.get("mean_C"),
            "M_B": m_result.get("mean_B"), "M_C": m_result.get("mean_C"),
            "Y_detail": y_result, "M_detail": m_result,
        })
        if (i + 1) % 10 == 0:
            print(f"  对比评测 {i+1}/{len(common)}", flush=True)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"对比结果保存至 {out_path}")
    return records


# ---- 中介聚合 ---------------------------------------------------------------

def aggregate_and_analyze(record_files: List[str], out_path: Optional[str] = None):
    all_recs = []
    for f in record_files:
        all_recs.extend(json.load(open(f, encoding="utf-8")))
    x = [r["x_level"] for r in all_recs]
    m = [float(r.get("M") or 0) for r in all_recs]
    y = [float(r["Y"]) for r in all_recs]
    g = [r["group"] for r in all_recs]

    res = run_mediation(x, m, y, g)
    report = res.summary()
    print(report)
    if out_path:
        Path(out_path).write_text(report, encoding="utf-8")
        print(f"\n中介报告已保存至 {out_path}")
    return res


# ---- 主入口 -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="recipe")
    ap.add_argument("--group", default="C", choices=["A", "B", "C"])
    ap.add_argument("--model")
    ap.add_argument("--test-data")
    ap.add_argument("--train-data", help="B/C 组构建 M_A 检索池用")
    ap.add_argument("--judges-config", default="configs/judges.yaml",
                    help="三裁判配置(DeepSeek-V4-Pro/Doubao-Seed-2-0-Lite/Qwen3.7-Plus)")
    ap.add_argument("--api-key", default="", help="M_A 仿真补充用")
    ap.add_argument("--ma-cache", default="", help="M_A 预生成缓存路径（命中零API等待）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--aggregate", nargs="+", help="聚合多组记录做中介分析")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compare", nargs=3, metavar=("FILE_A", "FILE_B", "FILE_C"),
                    help="对比+CoT评测：指定三组已生成的eval JSON，一次调用比较A/B/C")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_and_analyze(args.aggregate, args.out if args.out.endswith(".txt") else None)
        return

    if args.compare:
        from src.judge.judge_factory import build_judges
        judges = build_judges(args.domain, cfg_path=args.judges_config)
        if not judges:
            raise RuntimeError("未构建任何裁判，请在 configs/judges.yaml 填入 api_key")
        compare_evaluate(args.compare[0], args.compare[1], args.compare[2],
                         judges, args.out, args.limit)
        return

    if args.dry_run:
        _dry_run(args)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openai import OpenAI

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

    from src.judge.judge_factory import build_judges
    judges = build_judges(args.domain, cfg_path=args.judges_config)
    if not judges:
        raise RuntimeError("未构建任何裁判，请在 configs/judges.yaml 填入 api_key")

    ca, golden = None, None
    if args.group in ("B", "C"):
        llm = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com") if args.api_key else None
        pool = build_pool_from_dataset(args.train_data) if args.train_data else []
        ca = CounterAnswerer(retrieval_pool=pool, llm_client=llm, domain=args.domain,
                             cache_path=args.ma_cache or None)
    if args.group == "C":
        golden = GoldenQuestions(args.domain)

    ev = Evaluator(model, tok, args.domain, args.group, judges, ca, golden)
    test = json.load(open(args.test_data, encoding="utf-8"))
    savepoint = Path(args.out).with_suffix(".savepoint.json") if args.out else None
    records = ev.run(test, args.limit or None, str(savepoint) if savepoint else None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for r in records:
        r["_model"] = args.model
    json.dump(records, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ys = [r["Y"] for r in records if r["Y"] is not None]
    ms = [r["M"] for r in records if r["M"]]
    print(f"\n组 {args.group}: n={len(records)}, 平均Y={sum(ys)/len(ys):.3f}"
          + (f", 平均M={sum(ms)/len(ms):.3f}" if ms else ""))
    print(f"记录保存至 {args.out}")


def _dry_run(args):
    """用模拟评分跑通生成→评分→中介的整条流水线（无模型/无API）。"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    rng = random.Random(0)
    golden = GoldenQuestions(args.domain)
    sample_item = {"question": "赏析《登高》", "answer": "《登高》是杜甫晚年之作……",
                   "counter_question": "这首诗押什么韵？", "poem_title": "登高",
                   "poem_author": "杜甫", "poem_dynasty": "唐",
                   "dish_name": "宫保鸡丁"}

    print("=== 模拟三组生成 + 评分 ===")
    all_recs = []
    # 模拟真实中介机制：X 越大 -> M 越高；Y 由 M 驱动（真中介）+ 小直接效应
    m_base = {"A": 0, "B": 3.2, "C": 4.2}
    for grp in ["A", "B", "C"]:
        xv = X_LEVEL[Group(grp)]
        for _ in range(40):
            m_val = 0 if grp == "A" else max(1, min(5, rng.gauss(m_base[grp], 0.4)))
            # Y = 基础 + 0.6*M(中介主路径) + 0.15*X(直接效应) + 噪声
            #   A 组无反问，用一个低基础水平代表"未经反问深化"
            base = 2.4 if grp != "A" else 2.9
            y_val = base + 0.6 * m_val + 0.15 * xv + rng.gauss(0, 0.35)
            y_val = max(1, min(5, y_val))
            all_recs.append({"group": grp, "x_level": xv,
                             "M": round(m_val, 3), "Y": round(y_val, 3)})
        cq = None if grp == "A" else (
            sample_item["counter_question"] if grp == "B" else golden.pick(sample_item))
        print(f"  组{grp}: 反问示例={cq}")

    tmp = Path("results/_dryrun_records.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n=== 中介分析（模拟数据 n={len(all_recs)}）===")
    aggregate_and_analyze([str(tmp)])


if __name__ == "__main__":
    main()
