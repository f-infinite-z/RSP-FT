"""
菜谱QA两步生成v4（菜名清单→逐菜QA，覆盖八大菜系）。来源：DeepSeek API
用法：python gen_recipe_v4.py --api-key "sk-xxx" --all
开源重复性工程代码，由AI辅助快速落地，经人工检验验收通过。
"""
import json, random, sys, os, io, time
from pathlib import Path
from openai import OpenAI

CUISINES = ["川菜", "鲁菜", "粤菜", "苏菜", "闽菜", "浙菜", "湘菜", "徽菜"]

CUISINE_DESC = {
    "川菜": "麻辣鲜香，善用辣椒、花椒、豆瓣酱。代表菜：宫保鸡丁、麻婆豆腐、水煮鱼、回锅肉、鱼香肉丝",
    "鲁菜": "咸鲜为主，讲究火候和吊汤。代表菜：葱烧海参、糖醋鲤鱼、九转大肠、油焖大虾、四喜丸子",
    "粤菜": "清淡鲜美，注重食材本味。代表菜：白切鸡、清蒸鱼、煲仔饭、蜜汁叉烧、干炒牛河",
    "苏菜": "甜咸适中，刀工精细。代表菜：松鼠桂鱼、叫花鸡、狮子头、大煮干丝、无锡排骨",
    "闽菜": "鲜香清淡，善用高汤红糟。代表菜：佛跳墙、荔枝肉、醉排骨、沙茶面、海蛎煎",
    "浙菜": "清鲜脆嫩，讲究时令。代表菜：东坡肉、西湖醋鱼、龙井虾仁、宋嫂鱼羹、干炸响铃",
    "徽菜": "火候讲究，善用火腿山珍。代表菜：臭鳜鱼、毛豆腐、一品锅、李鸿章大杂烩、黄山炖鸽",
    "湘菜": "香辣酸爽，油重色浓。代表菜：剁椒鱼头、小炒肉、口味虾、毛氏红烧肉、东安子鸡",
}

QA_TYPES = ["cooking_method", "ingredient_choice", "flavor_technique", "cuisine_culture", "nutrition_tips", "home_cooking", "comparison"]

SYSTEM_DISHES = "你是中华菜谱专家。请列出指定菜系的菜品名称，每道菜一行。只输出菜名列表，不要解释。"
SYSTEM_QA = "你是中华菜谱专家。为给定菜品生成一个问答对。输出JSON：{\"question\":\"...\",\"answer\":\"...\",\"qa_type\":\"...\",\"counter_question\":\"...\",\"counter_answer\":\"...\"}。只输出JSON。"

API_KEY = ""
client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    return client


def fetch_dish_names(cuisine, n):
    """为菜系生成菜名列表"""
    prompt = f"请列出{cuisine}（{CUISINE_DESC[cuisine]}）的{n}道不同的真实菜品名称。尽量涵盖热菜、凉菜、小吃、汤羹、面点等各品类。每行一个菜名，只输出菜名。"
    try:
        resp = get_client().chat.completions.create(
            model="deepseek-chat", temperature=0.9, max_tokens=4000,
            messages=[{"role": "system", "content": SYSTEM_DISHES},
                       {"role": "user", "content": prompt}]
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        dishes = [l.strip().lstrip("0123456789.、-）) ").strip() for l in lines if l.strip()]
        return dishes
    except Exception as e:
        print(f"    E: {e}", flush=True)
        return []


def gen_qa(dish_name, cuisine, qa_type):
    """为单个菜品生成QA"""
    t_desc = {"cooking_method":"详细做法步骤与关键技巧","ingredient_choice":"食材选择、替代、搭配建议","flavor_technique":"口味调配与烹饪技法","cuisine_culture":"菜名由来、文化典故、地域特色","nutrition_tips":"营养知识、食用禁忌、健康提示","home_cooking":"家庭简化做法、省时省力技巧","comparison":"与其他菜品的异同对比分析"}
    prompt = f"菜品：{dish_name}（{cuisine}）\n题型：{qa_type}——{t_desc.get(qa_type, '')}\n请生成一个专业准确的问答对，答案要详细。"
    try:
        resp = get_client().chat.completions.create(
            model="deepseek-chat", temperature=0.8, max_tokens=2000,
            messages=[{"role": "system", "content": SYSTEM_QA},
                       {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content.strip())
        data["dish_name"] = dish_name
        data["cuisine"] = cuisine
        data["qa_type"] = qa_type
        return data
    except:
        return None


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def gen_cuisine(cuisine, n_target, base_dir):
    path = f"{base_dir}/{cuisine}.json"
    data = load_json(path)
    existing = {d["dish_name"] for d in data}
    print(f"\n=== {cuisine}: {len(data)}/{n_target} ===", flush=True)

    while len(data) < n_target:
        # 如果需要更多菜名，先获取
        if not existing or len(data) >= len(existing) * len(QA_TYPES) * 0.9:
            n_names = min(300, n_target * 2)
            print(f"  获取菜名列表 (目标{n_names})...", flush=True)
            new_names = fetch_dish_names(cuisine, n_names)
            before = len(existing)
            existing.update(new_names)
            print(f"  +{len(existing) - before} 新菜名 (总计{len(existing)})", flush=True)

        # 为每个菜名生成QA
        batch = list(existing)
        random.shuffle(batch)
        for name in batch:
            if len(data) >= n_target:
                break
            qa_type = QA_TYPES[len(data) % len(QA_TYPES)]
            qa = gen_qa(name, cuisine, qa_type)
            if qa and qa.get("question") and qa.get("answer"):
                data.append(qa)
                if len(data) % 50 == 0:
                    print(f"  [{len(data)}/{n_target}] {qa_type}: {name}", flush=True)
                    save_json(data, path)
            time.sleep(0.1)

        save_json(data, path)
    return data


def main():
    global API_KEY
    API_KEY = sys.argv[sys.argv.index("--api-key") + 1] if "--api-key" in sys.argv else os.getenv("DEEPSEEK_API_KEY")
    base_dir = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "data/recipe/qa_pairs/by_cuisine"
    n_per = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 750

    if "--all" in sys.argv:
        cuisines = CUISINES
    elif "--cuisine" in sys.argv:
        cuisines = [sys.argv[sys.argv.index("--cuisine") + 1]]
    else:
        print("用法: --cuisine '川菜' 或 --all"); return

    for c in cuisines:
        gen_cuisine(c, n_per, base_dir)

    if len(cuisines) == 8:
        all_data = []
        for c in CUISINES:
            all_data.extend(load_json(f"{base_dir}/{c}.json"))
        random.shuffle(all_data)
        parent = Path(base_dir).parent
        save_json(all_data[:5000], parent / "train.json")
        save_json(all_data[5000:5500], parent / "val.json")
        save_json(all_data[5500:6000], parent / "test.json")
        print(f"\n=== ALL DONE: {len(all_data)} -> train=5000/val=500/test=500 ===")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
