"""
古诗词 QA 上下文修复脚本

本脚本由 AI 辅助生成初稿，经人工适配、调试、校验。
核心算法与实验设计由作者主导完成。问题：约 75% 的 question 脱离上下文（如"这首诗表达了诗人怎样的情感？"重复出现），
      训练与裁判都无法定位是哪首诗。

修复：从 chinese-poetry 原始数据按 (author, title) 反查正文，
      按题型把「诗题/作者/朝代/正文」智能注入 question 开头。
      answer / counter_question / counter_answer 保持不变。

题型注入策略：
  - line_completion  : 已含诗题+作者，仅确保有正文可依据 -> 附正文（若原问未含）
  - author_id        : 需正文才能判断作者 -> 注入正文（去掉答案泄露的作者名不适用，问题本就问作者）
  - dynasty_judge    : 多数已含诗题+作者 -> 若缺则补
  - 其他(情感/意象/修辞/对比): 注入「诗题+作者+正文」

幂等：已含诗题的 question 跳过注入（避免重复运行叠加）。

用法：
  python -m src.data.fix_poetry_context --dir data/poetry/qa_pairs          # 就地修复 train/val/test（自动备份）
  python -m src.data.fix_poetry_context --file data/poetry/qa_pairs/train.json --dry-run
"""
import argparse
import io
import json
import shutil
import sys
from pathlib import Path


def build_index():
    """(author, title) -> full_text"""
    from src.data.poetry_processor import load_all_poetry
    poems = load_all_poetry()
    idx = {}
    for p in poems:
        key = (p.author, p.title)
        if key not in idx and p.full_text.strip():
            idx[key] = p.full_text.strip()
    return idx


# 需要完整正文才能作答/评测的题型
NEEDS_FULLTEXT = {"author_id", "emotion_analysis", "imagery_analysis",
                  "rhetoric_analysis", "comparison", "line_completion"}


def make_context(item, full_text):
    """按题型构造注入前缀。"""
    author = item.get("poem_author", "")
    title = item.get("poem_title", "")
    dynasty = item.get("poem_dynasty", "")
    qa_type = item.get("qa_type", "")

    head = f"《{title}》"
    if dynasty and author:
        head += f"（{dynasty}·{author}）"
    elif author:
        head += f"（{author}）"

    if qa_type == "author_id":
        # 问作者：给正文，不能在前缀泄露作者
        return f"阅读下面这首诗：\n{full_text}\n请问：", True
    if qa_type == "dynasty_judge":
        # 问朝代：可给诗题+作者+正文
        return f"{head}原文如下：\n{full_text}\n请问：", False
    if qa_type == "line_completion":
        # 续写：题干本身含残句，补充诗题作者即可，不重复正文（避免泄露答案）
        return f"关于{head}，", False
    # 情感/意象/修辞/对比：给完整语境
    return f"阅读{head}：\n{full_text}\n请问：", False


def already_has_context(item):
    """幂等判断：question 是否已含诗题或正文标志。"""
    q = item.get("question", "")
    title = item.get("poem_title", "")
    return (title and title in q) or ("原文如下" in q) or ("阅读下面这首诗" in q) or q.startswith("阅读《")


def fix_item(item, idx):
    """返回 (是否修改, 修改后的item)。"""
    if already_has_context(item):
        return False, item
    key = (item.get("poem_author"), item.get("poem_title"))
    full_text = idx.get(key)
    if not full_text:
        return False, item
    prefix, _ = make_context(item, full_text)
    old_q = item["question"]
    # author_id 的原问通常是"这首诗的作者是谁？"，直接拼
    item["question"] = prefix + old_q
    item["_context_injected"] = True
    return True, item


def process_file(path, idx, dry_run=False):
    data = json.load(open(path, encoding="utf-8"))
    changed = 0
    samples = []
    for i, item in enumerate(data):
        did, data[i] = fix_item(item, idx)
        if did:
            changed += 1
            if len(samples) < 3:
                samples.append(data[i]["question"][:100])
    print(f"  {Path(path).name}: {len(data)}条, 注入上下文 {changed} ({100*changed/len(data):.0f}%)")
    for s in samples:
        print(f"      例: {s}")
    if not dry_run and changed:
        backup = str(path) + ".bak"
        if not Path(backup).exists():
            shutil.copy(path, backup)
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("构建正文索引…", flush=True)
    idx = build_index()

    files = []
    if args.dir:
        files = [str(p) for p in Path(args.dir).glob("*.json") if not p.name.endswith(".bak")]
    elif args.file:
        files = [args.file]
    else:
        print("需指定 --dir 或 --file")
        return

    total = 0
    for f in files:
        total += process_file(f, idx, args.dry_run)
    mode = "（dry-run，未写入）" if args.dry_run else "（已写入，原文件备份为 .bak）"
    print(f"\n完成：共注入 {total} 条 {mode}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
