"""
古诗词QA对生成器
基于 chinese-poetry 数据 + LLM API，自动生成训练用的QA对和反问数据。

本脚本由 AI 辅助生成初稿，经人工适配、调试、校验。
原创 RSP-FT 训练代码为作者独立实现（见 src/training/）。
数据源：chinese-poetry (https://github.com/chinese-poetry/chinese-poetry)。
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

from poetry_processor import (load_all_poetry, PoetryEntry,
                               filter_by_author, FAMOUS_TANG, FAMOUS_SONG)

QA_TYPES = {
    "author_id": "根据给出的诗句，判断作者是谁",
    "line_completion": "给出上句/下句，补全诗句",
    "dynasty_judge": "判断这首诗的朝代",
    "theme_tag": "这首诗属于什么主题/风格",
    "title_lookup": "根据诗题找出对应的全文",
    "full_to_title": "根据全文反推诗题",
    "imagery_analysis": "分析诗中某意象的含义",
    "emotion_analysis": "分析诗中的情感基调",
    "rhetoric_analysis": "分析诗中的修辞手法",
    "poet_bio": "介绍诗人某方面的生平或风格",
    "comparison": "将这首诗与另一首进行对比",
    "rhythmic_judge": "判断词牌名或格律特点",
}


SYSTEM_PROMPT = """你是一个专业的QA对生成器。你必须严格按照JSON格式输出，不得有任何多余文字。

输出格式如下（这是唯一的合法格式）：
{"question":"...", "answer":"...", "qa_type":"...", "counter_question":"...", "counter_answer":"..."}

要求：
- question: 基于给定诗词提出的问题，自然流畅
- answer: 详尽准确的回答，引用诗文原文支撑观点
- qa_type: 题型标签（author_id/line_completion/emotion_analysis/imagery_analysis/rhetoric_analysis/comparison/dynasty_judge之一）
- counter_question: 一个针对answer内容的追问，用于检验回答深度
- counter_answer: 反问答的参考答案
- 回答中如有双引号，使用\\"转义
- 只输出一行JSON，不输出任何其他内容"""


def build_prompt_for_poem(poem: PoetryEntry, qa_type: str) -> str:
    """为指定诗词和题型构建生成prompt"""
    full_text = poem.full_text

    type_descriptions = {
        "author_id": f'请根据以下诗词内容生成一个"判断作者"的问答。不要直接给出诗题和全文，而是挑选其中最有辨识度的一两句诗作为问题，问作者是谁。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "line_completion": f'请生成一个"名句续写"的问答。挑选诗中最有名的1-2句作为上半部分，问题是"请写出下一句/补全这首诗"。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "dynasty_judge": f'请生成一个"朝代判断"的问答。选取诗中能反映时代特色的诗句，问这首诗可能是哪个朝代的，并说明判断依据。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "emotion_analysis": f'请生成一个"情感分析"的问答。选诗中最能体现情感的诗句，问这首诗表达了什么情感，请结合具体语句和创作背景分析。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "imagery_analysis": f'请生成一个"意象分析"的问答。选诗中1-2个关键意象（如月、柳、酒、笛、雁等），问该意象在诗中象征什么。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "rhetoric_analysis": f'请生成一个"修辞手法分析"的问答。选诗中使用了明显修辞手法的诗句，分析其手法和效果。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
        "comparison": f'请生成一个"对比分析"的问答。将这首诗与{poem.author}的另一首风格相近或相反的作品进行对比（你可以自由选择对比对象），分析异同。\n\n诗词信息：\n作者：{poem.author}\n朝代：{poem.dynasty}\n诗题：{poem.title}\n全文：\n{full_text}',
    }

    return type_descriptions.get(qa_type, type_descriptions["emotion_analysis"])


def extract_json(raw: str) -> str:
    """从LLM输出中提取JSON，处理常见格式问题"""
    raw = raw.strip()
    # 去掉markdown代码块
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    # 找到第一个{和最后一个}
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return raw.strip()


def generate_qa(poem: PoetryEntry, qa_type: str,
                client: OpenAI, model: str = "deepseek-chat") -> Optional[Dict]:
    """对单首诗词生成一个QA对"""
    prompt = build_prompt_for_poem(poem, qa_type)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            raw = resp.choices[0].message.content.strip()
            json_str = extract_json(raw)
            return json.loads(json_str)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < 2:
                continue
            safe_title = poem.title.encode('ascii', errors='replace').decode('ascii')
            print(f"  FAIL [{qa_type}] {poem.author}: {safe_title}")
            return None


def generate_qa_batch(poems: List[PoetryEntry], n_total: int = 1000,
                      output_path: str = "data/poetry/qa_pairs/train.json",
                      model: str = "deepseek-chat",
                      api_key: str = None,
                      base_url: str = "https://api.deepseek.com") -> List[Dict]:
    """
    批量生成QA对。

    用法：
      python qa_generator.py --api-key YOUR_DEEPSEEK_KEY --n 1000

    输出: data/poetry/qa_pairs/train.json
    """
    client = OpenAI(api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                    base_url=base_url)

    # 分配题型
    type_distribution = {
        "author_id": 0.15, "line_completion": 0.15, "dynasty_judge": 0.08,
        "emotion_analysis": 0.18, "imagery_analysis": 0.18,
        "rhetoric_analysis": 0.14, "comparison": 0.12
    }

    qa_pairs = []
    failed = 0
    # 按比例分配，但保证每种至少1条（如总数足够）
    n_per_type = {}
    remaining = n_total
    types = list(type_distribution.keys())
    for i, t in enumerate(types):
        if i == len(types) - 1:
            n_per_type[t] = remaining
        else:
            n = max(1, int(n_total * type_distribution[t]))
            n_per_type[t] = min(n, remaining)
            remaining -= n_per_type[t]
    n_per_type = {t: n for t, n in n_per_type.items() if n > 0}

    for qa_type, target_n in n_per_type.items():
        type_poems = list(poems)
        random.shuffle(type_poems)
        generated = 0

        for poem in type_poems:
            if generated >= target_n:
                break
            if not poem.full_text.strip():
                continue

            result = generate_qa(poem, qa_type, client, model)
            if result:
                result["poem_author"] = poem.author
                result["poem_title"] = poem.title
                result["poem_dynasty"] = poem.dynasty
                qa_pairs.append(result)
                if len(qa_pairs) % 100 == 0:
                    print(f"  Generated {len(qa_pairs)}/{n_total} QA pairs...")
            else:
                failed += 1

    # 保存
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(qa_pairs)} QA pairs saved to {output_path}")
    print(f"Failed: {failed}")
    return qa_pairs


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="DeepSeek/OpenAI API key")
    parser.add_argument("--n", type=int, default=1000, help="Number of QA pairs to generate")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output", default="data/poetry/qa_pairs/train.json")
    parser.add_argument("--famous-only", action="store_true")
    args = parser.parse_args()

    print("Loading poetry data...")
    all_poems = load_all_poetry()
    print(f"Loaded {len(all_poems)} poems total")

    if args.famous_only:
        all_poems = filter_by_author(all_poems, FAMOUS_TANG + FAMOUS_SONG)
        print(f"Filtered to {len(all_poems)} poems by famous authors")

    generate_qa_batch(all_poems, n_total=args.n, model=args.model,
                      api_key=args.api_key, output_path=args.output)
