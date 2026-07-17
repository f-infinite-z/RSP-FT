"""
古诗词QA对生成器v2（增量追加+断点续传+自动split）。来源：chinese-poetry+DeepSeek API
用法：python gen_qa.py --api-key "sk-xxx" --n 500 --output data/poetry/qa_pairs/train.json
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, random, sys, os, io
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.poetry_processor import load_all_poetry, filter_by_author, FAMOUS_TANG, FAMOUS_SONG

SYSTEM_MSG = """你是古诗词QA对生成器。给定一首诗及其作者、朝代，生成一个JSON格式的QA对。

JSON必须包含以下字段（中文）：
{"question": "关于这首诗的问题",
 "answer": "准确详细的回答",
 "qa_type": "题型标签",
 "counter_question": "针对回答内容的追问",
 "counter_answer": "对追问的参考答案"}

qa_type可选值：author_id, line_completion, emotion_analysis, imagery_analysis, rhetoric_analysis, comparison, dynasty_judge"""


def gen_one(client, poem, qa_type):
    user_prompt = f"""作者：{poem.author}
朝代：{poem.dynasty}
诗题：{poem.title}
全文：
{poem.full_text}

请为这首诗生成一个题型为"{qa_type}"的QA对。只输出JSON，不要任何解释。"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM_MSG},
                       {"role": "user", "content": user_prompt}],
            temperature=0.7, max_tokens=1500,
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except:
        return None


def load_existing(path):
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    api_key = sys.argv[sys.argv.index("--api-key") + 1] if "--api-key" in sys.argv else os.getenv("DEEPSEEK_API_KEY")
    n_target = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 500
    out_path = sys.argv[sys.argv.index("--output") + 1] if "--output" in sys.argv else "data/poetry/qa_pairs/train.json"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 恢复已有数据
    qa_pairs = load_existing(out_path)
    existing_poems = {(q["poem_author"], q["poem_title"]) for q in qa_pairs
                       if "poem_author" in q and "poem_title" in q}

    if qa_pairs:
        print(f"Resuming from {len(qa_pairs)} existing QA pairs, target: {n_target}")

    # 加载诗词
    all_poems = load_all_poetry()
    poems = filter_by_author(all_poems, FAMOUS_TANG + FAMOUS_SONG)
    poems = [p for p in poems if (p.author, p.title) not in existing_poems
             and p.full_text.strip() and len(p.paragraphs) >= 2]
    random.shuffle(poems)

    types = ["emotion_analysis", "imagery_analysis", "rhetoric_analysis",
             "author_id", "line_completion", "comparison", "dynasty_judge"]

    idx = 0
    success = len(qa_pairs)
    fail = 0

    while success < n_target and idx < len(poems) - 1:
        poem = poems[idx]
        qa_type = types[idx % len(types)]
        idx += 1

        result = gen_one(client, poem, qa_type)
        if result and all(k in result for k in ["question", "answer"]):
            result["poem_author"] = poem.author
            result["poem_title"] = poem.title
            result["poem_dynasty"] = poem.dynasty
            result["qa_type"] = qa_type
            qa_pairs.append(result)
            success += 1
            print(f"  [{success}/{n_target}] {qa_type}: {poem.author}", flush=True)

            if success % 50 == 0:
                save_json(qa_pairs, out_path)
                print(f"    -> saved {success} pairs", flush=True)
        else:
            fail += 1

    save_json(qa_pairs, out_path)

    # 分割train/val/test
    random.shuffle(qa_pairs)
    total = len(qa_pairs)
    t80 = int(total * 0.8)
    t90 = int(total * 0.9)
    parent = Path(out_path).parent
    save_json(qa_pairs[:t80], parent / "train.json")
    save_json(qa_pairs[t80:t90], parent / "val.json")
    save_json(qa_pairs[t90:], parent / "test.json")

    print(f"\nDone: {success} total ({fail} failed)")
    print(f"Split: train={t80}, val={t90-t80}, test={total-t90}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
