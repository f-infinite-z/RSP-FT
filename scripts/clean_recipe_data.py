"""
菜谱三裁判自洁（factual<6/分歧>2双重过滤+并发+断点+熔断）。来源：DeepSeek/Doubao/Qwen API

缺陷覆盖：
  1. 双重过滤：avg_factual < 6 OR 裁判间 max_diff > 2
  2. 空值容错：至少 2 个裁判有效才计算均值
  3. 阈值 6（保守，避免轻度编造漏检）
  4. 反问 CQ 同步校验

用法：
  python clean_recipe_data.py --judges-config configs/judges.yaml --source train --limit 0
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json
import argparse
import os
import sys
import time
import signal
import shutil
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 全局引用，供 signal handler 保存
_SAFEGUARD_STATE = None


def _save_checkpoint(out_dir, label, clean, flagged, failed, stats):
    """原子写入 + 保留 .bak 备份，防止写入中断损坏或误删。"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in [("clean", clean), ("flagged", flagged), ("failed", failed)]:
        path = Path(out_dir) / f"{label}_{name}.json"
        tmp = Path(out_dir) / f"{label}_{name}.json.tmp"
        json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if path.exists():
            shutil.copy2(path, Path(out_dir) / f"{label}_{name}.json.bak")
        tmp.replace(path)
    stats_path = Path(out_dir) / f"{label}_stats.json"
    tmp_stats = Path(out_dir) / f"{label}_stats.json.tmp"
    json.dump(stats, open(tmp_stats, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if stats_path.exists():
        shutil.copy2(stats_path, Path(out_dir) / f"{label}_stats.json.bak")
    tmp_stats.replace(stats_path)


def _safeguard_handler(signum, frame):
    """SIGINT/SIGTERM 时触发最后一次保存。"""
    if _SAFEGUARD_STATE:
        clean, flagged, failed, stats, out_dir, label = _SAFEGUARD_STATE
        print(f"\n  [safeguard] 收到中断信号，正在保存... clean={len(clean)} flagged={len(flagged)}", flush=True)
        _save_checkpoint(out_dir, label, clean, flagged, failed, stats)
        print("  [safeguard] 已保存，可安全退出", flush=True)
    os._exit(0)

from src.judge.judge_factory import build_judges
from src.judge.llm_judge import Judge


def score_single(judges: List[Judge], method: str, min_judges: int = 2, **kwargs) -> Tuple[Optional[float], Dict]:
    """
    调用多裁判评分，返回 (均值, 逐裁判详情)。
    method: "answer" | "cq"
    失败返回 (None, {"errors": n, "judges_valid": n})
    """
    scores = {}
    errors = 0
    for j in judges:
        name = j.name or j.model
        try:
            if method == "answer":
                s = j.score_answer(kwargs["question"], kwargs["answer"])
            else:
                s = j.score_cq(kwargs["question"], kwargs["counter_question"])
            scores[name] = s.factual if s else None
            if s is None:
                errors += 1
        except Exception:
            scores[name] = None
            errors += 1

    valid_vals = [v for v in scores.values() if v is not None]
    n_valid = len(valid_vals)

    if n_valid < min_judges:
        return None, {"judges_valid": n_valid, "errors": errors, "per_judge": scores}

    avg = sum(valid_vals) / n_valid
    max_diff = max(valid_vals) - min(valid_vals) if n_valid >= 2 else 0.0
    return avg, {
        "judges_valid": n_valid,
        "errors": errors,
        "avg": round(avg, 2),
        "max_diff": round(max_diff, 2),
        "disagreement": max_diff > 2.0,
        "per_judge": scores,
    }


def _process_one(item: Dict, judges: List[Judge], min_judges: int, fact_threshold: float, skip_cq: bool) -> Tuple[Dict, str]:
    """处理单条数据，返回 (record, category) where category in {clean, flagged, failed}."""
    q = item.get("question", "")
    a = item.get("answer", "")
    cq = item.get("counter_question", "")

    a_avg, a_detail = score_single(judges, "answer", min_judges=min_judges, question=q, answer=a)

    cq_avg, cq_detail = None, None
    if not skip_cq and cq and cq.strip():
        cq_avg, cq_detail = score_single(judges, "cq", min_judges=min_judges, question=q, counter_question=cq)

    reasons = []
    if a_avg is None:
        reasons.append("insufficient_judges")
    else:
        if a_avg < fact_threshold:
            reasons.append(f"answer_low_fact({a_avg:.1f})")
        if a_detail.get("disagreement"):
            reasons.append(f"answer_disagree(max_diff={a_detail['max_diff']})")

    if cq_avg is not None and cq_avg < fact_threshold:
        reasons.append(f"cq_low_fact({cq_avg:.1f})")

    record = {**item, "answer_fact_score": a_avg, "answer_detail": a_detail,
              "cq_fact_score": cq_avg, "cq_detail": cq_detail, "flag_reasons": reasons}

    if not reasons:
        return record, "clean"
    elif "insufficient_judges" in reasons and len(reasons) == 1:
        return record, "failed"
    else:
        return record, "flagged"


def _load_progress(out_dir: str, label: str, items: List[Dict]) -> Tuple[List, List, List, set]:
    """加载已有进度，返回 (clean, flagged, failed, done_questions)。"""
    clean, flagged, failed = [], [], []
    done = set()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, lst in [("clean", clean), ("flagged", flagged), ("failed", failed)]:
        path = Path(out_dir) / f"{label}_{name}.json"
        if path.exists():
            loaded = json.load(open(path, "r", encoding="utf-8"))
            lst.extend(loaded)
            done.update(r.get("question", "") for r in loaded)
    if done:
        print(f"  [resume] 从断点恢复，已完成 {len(done)}/{len(items)}，跳过已处理条目")
    return clean, flagged, failed, done


def clean_dataset(data: List[Dict], judges: List[Judge], label: str = "",
                  out_dir: str = "data/recipe/cleaned",
                  fact_threshold: float = 6.0, min_judges: int = 2,
                  skip_cq: bool = False, workers: int = 1,
                  limit: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    批量自洁。返回 (clean, flagged, failed)。
    workers: 并发线程数（默认1=串行，建议2-3）。
    """
    items = data[:limit] if limit > 0 else data
    clean, flagged, failed, done_questions = _load_progress(out_dir, label, items)
    stats = {"answer_low_fact": 0, "answer_disagree": 0,
             "cq_low_fact": 0, "insufficient_judges": 0, "total": len(items)}

    # 恢复 stats：从已有记录重新统计
    for lst in (clean, flagged, failed):
        for r in lst:
            _update_stats(stats, r)

    # 过滤已处理条目
    remaining = [item for item in items if item.get("question", "") not in done_questions]
    n = len(items)
    processed = len(done_questions)
    if not remaining:
        print(f"  [{label}] 全部完成 {processed}/{n}")
        _save_checkpoint(out_dir, label, clean, flagged, failed, stats)
        _print_summary(label, stats, clean, flagged, failed)
        return clean, flagged, failed

    global _SAFEGUARD_STATE
    _SAFEGUARD_STATE = (clean, flagged, failed, stats, out_dir, label)
    signal.signal(signal.SIGINT, _safeguard_handler)
    signal.signal(signal.SIGTERM, _safeguard_handler)

    if workers <= 1:
        # 串行
        for i, item in enumerate(remaining, start=processed):
            record, cat = _process_one(item, judges, min_judges, fact_threshold, skip_cq)
            _update_stats(stats, record)
            {"clean": clean, "flagged": flagged, "failed": failed}[cat].append(record)
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{n}  clean={len(clean)} flagged={len(flagged)} failed={len(failed)}", flush=True)
                _save_checkpoint(out_dir, label, clean, flagged, failed, stats)
    else:
        # 并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, item, judges, min_judges, fact_threshold, skip_cq): item
                       for item in remaining}
            for fut in concurrent.futures.as_completed(futures):
                record, cat = fut.result()
                _update_stats(stats, record)
                {"clean": clean, "flagged": flagged, "failed": failed}[cat].append(record)
                done = len(clean) + len(flagged) + len(failed)
                if done % 50 == 0:
                    print(f"  [{label}] {done}/{n}  clean={len(clean)} flagged={len(flagged)} failed={len(failed)}", flush=True)
                    _save_checkpoint(out_dir, label, clean, flagged, failed, stats)

    _save_checkpoint(out_dir, label, clean, flagged, failed, stats)
    _print_summary(label, stats, clean, flagged, failed)
    return clean, flagged, failed


def _update_stats(stats: Dict, record: Dict):
    for reason in record.get("flag_reasons", []):
        if "insufficient" in reason:
            stats["insufficient_judges"] += 1
        elif "answer_low_fact" in reason:
            stats["answer_low_fact"] += 1
        elif "answer_disagree" in reason:
            stats["answer_disagree"] += 1
        elif "cq_low_fact" in reason:
            stats["cq_low_fact"] += 1


def _save_checkpoint(out_dir, label, clean, flagged, failed, stats):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in [("clean", clean), ("flagged", flagged), ("failed", failed)]:
        path = Path(out_dir) / f"{label}_{name}.json"
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    stats_path = Path(out_dir) / f"{label}_stats.json"
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _print_summary(label, stats, clean, flagged, failed):
    n = stats["total"]
    print(f"\n{'='*50}")
    print(f"  [{label}] 自洁完成：{n} 条")
    print(f"  clean:   {len(clean):>5} ({len(clean)/n*100:.1f}%)")
    print(f"  flagged: {len(flagged):>5} ({len(flagged)/n*100:.1f}%)")
    print(f"  failed:  {len(failed):>5} ({len(failed)/n*100:.1f}%)")
    print(f"  标记原因分布：")
    print(f"    回答事实分<6:         {stats['answer_low_fact']}")
    print(f"    裁判分歧>2:           {stats['answer_disagree']}")
    print(f"    反问CQ事实分<6:       {stats['cq_low_fact']}")
    print(f"    有效裁判不足2:        {stats['insufficient_judges']}")
    print(f"{'='*50}\n")


def merge_clean_splits(clean_data: List[Dict], output_dir: str):
    """从清洗后的数据中重新分割 train/val/test (5000/500/500)。"""
    import random
    random.seed(42)
    shuffled = clean_data[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = min(5000, int(n * 0.85))
    val_end = min(train_end + 500, int(n * 0.92))
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:val_end + 500]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.json"
        json.dump(subset, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  {name}: {len(subset)} 条 → {path}")


def main():
    ap = argparse.ArgumentParser(description="菜谱数据三裁判自洁")
    ap.add_argument("--judges-config", default="configs/judges.yaml")
    ap.add_argument("--source", default="train", choices=["train", "val", "test", "all"],
                    help="清洗哪个分片（all=三个都洗）")
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="事实分阈值，低于此值标记可疑（默认6.0）")
    ap.add_argument("--limit", type=int, default=0, help="限制条数（0=全部）")
    ap.add_argument("--out-dir", default="data/recipe/cleaned")
    ap.add_argument("--min-judges", type=int, default=2,
                    help="最少有效裁判数（默认2，单裁判无分歧检测）")
    ap.add_argument("--skip-cq", action="store_true", help="跳过反问CQ事实校验（省50%时间，cq_low_fact历史为0）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发线程数（默认1串行，建议2-3）")
    ap.add_argument("--merge", action="store_true", help="全部清洗后重新合并 train/val/test")
    ap.add_argument("--dry-run", action="store_true", help="仅统计不实际调用API")
    args = ap.parse_args()

    if args.dry_run:
        for split in (["train", "val", "test"] if args.source == "all" else [args.source]):
            path = f"data/recipe/qa_pairs/{split}.json"
            if Path(path).exists():
                data = json.load(open(path, encoding="utf-8"))
                print(f"  {split}: {len(data)} 条")
        return

    judges = build_judges("recipe", cfg_path=args.judges_config)
    if not judges:
        raise RuntimeError("未构建任何裁判，请在 configs/judges.yaml 填入 api_key")
    print(f"已加载 {len(judges)} 个裁判，阈值={args.threshold}")

    all_clean = []
    splits = ["train", "val", "test"] if args.source == "all" else [args.source]

    for split in splits:
        path = f"data/recipe/qa_pairs/{split}.json"
        if not Path(path).exists():
            print(f"  跳过 {split}：文件不存在")
            continue
        data = json.load(open(path, encoding="utf-8"))
        clean, flagged, failed = clean_dataset(
            data, judges, label=split, out_dir=args.out_dir,
            fact_threshold=args.threshold, min_judges=args.min_judges,
            skip_cq=args.skip_cq, workers=args.workers, limit=args.limit)
        all_clean.extend(clean)

    if args.merge and all_clean:
        merge_clean_splits(all_clean, "data/recipe/qa_pairs_clean")


if __name__ == "__main__":
    main()
