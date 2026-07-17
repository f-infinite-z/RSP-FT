"""
古诗词数据处理模块
基于 chinese-poetry 数据集的结构分析结果，加载并标准化全唐诗+宋词数据。

本脚本由 AI 辅助生成初稿，经人工适配、调试、校验。
数据源：chinese-poetry (https://github.com/chinese-poetry/chinese-poetry)。
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

CHINESE_POETRY_DIR = Path(__file__).parent.parent.parent.parent / "chinese-poetry"


class PoetryEntry:
    """标准化诗词条目"""
    def __init__(self, author: str, title: str, dynasty: str, paragraphs: List[str],
                 tags: List[str] = None, rhythmic: str = None, uid: str = None):
        self.author = author
        self.title = title
        self.dynasty = dynasty
        self.paragraphs = paragraphs
        self.tags = tags or []
        self.rhythmic = rhythmic
        self.uid = uid

    @property
    def full_text(self) -> str:
        return "\n".join(self.paragraphs)

    @property
    def first_line(self) -> str:
        return self.paragraphs[0] if self.paragraphs else ""

    def to_dict(self) -> dict:
        return {
            "author": self.author, "title": self.title, "dynasty": self.dynasty,
            "paragraphs": self.paragraphs, "tags": self.tags,
            "rhythmic": self.rhythmic, "uid": self.uid
        }


def load_tang_poetry() -> List[PoetryEntry]:
    """加载全唐诗/poet.tang (唐代诗歌, ~58,000首)"""
    entries = []
    tang_dir = CHINESE_POETRY_DIR / "全唐诗"
    for fname in sorted(os.listdir(tang_dir)):
        if not fname.startswith("poet.tang."):
            continue
        with open(tang_dir / fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entries.append(PoetryEntry(
                author=item.get("author", "佚名"),
                title=item.get("title", "无题"),
                dynasty="唐",
                paragraphs=item.get("paragraphs", []),
                tags=item.get("tags", []),
                uid=item.get("id")
            ))
    return entries


def load_song_poetry() -> List[PoetryEntry]:
    """加载全唐诗/poet.song (宋代诗歌, ~255,000首)"""
    entries = []
    song_dir = CHINESE_POETRY_DIR / "全唐诗"
    for fname in sorted(os.listdir(song_dir)):
        if not fname.startswith("poet.song."):
            continue
        with open(song_dir / fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entries.append(PoetryEntry(
                author=item.get("author", "佚名"),
                title=item.get("title", "无题"),
                dynasty="宋",
                paragraphs=item.get("paragraphs", []),
                tags=item.get("tags", []),
                uid=item.get("id")
            ))
    return entries


def load_song_ci() -> List[PoetryEntry]:
    """加载宋词 (宋代词, ~21,053首)"""
    entries = []
    ci_dir = CHINESE_POETRY_DIR / "宋词"
    for fname in sorted(os.listdir(ci_dir)):
        if not fname.startswith("ci.song."):
            continue
        with open(ci_dir / fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            rhythmic = item.get("rhythmic", "")
            title = rhythmic if rhythmic else "无题"
            entries.append(PoetryEntry(
                author=item.get("author", "佚名"),
                title=title,
                dynasty="宋",
                paragraphs=item.get("paragraphs", []),
                rhythmic=rhythmic,
                tags=item.get("tags", [])
            ))
    return entries


def load_all_poetry() -> List[PoetryEntry]:
    """加载所有诗词数据"""
    entries = load_tang_poetry() + load_song_poetry() + load_song_ci()
    print(f"Loaded {len(entries)} poems: "
          f"Tang poetry={len(load_tang_poetry())}, "
          f"Song poetry={len(load_song_poetry())}, "
          f"Song ci={len(load_song_ci())}")
    return entries


def filter_by_author(entries: List[PoetryEntry], authors: List[str]) -> List[PoetryEntry]:
    """按作者筛"""
    return [e for e in entries if e.author in authors]


def filter_by_tag(entries: List[PoetryEntry], tags: List[str]) -> List[PoetryEntry]:
    """按标签筛选（覆盖率仅2-6%，仅对有名篇有效）"""
    return [e for e in entries if any(t in e.tags for t in tags)]


def sample_poems(entries: List[PoetryEntry], n: int, min_lines: int = 2) -> List[PoetryEntry]:
    """随机采样n首，保证至少min_lines行"""
    pool = [e for e in entries if len(e.paragraphs) >= min_lines]
    return random.sample(pool, min(n, len(pool)))


def get_author_stats(entries: List[PoetryEntry]) -> Dict[str, int]:
    """作者-作品数统计"""
    stats = {}
    for e in entries:
        stats[e.author] = stats.get(e.author, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: -x[1]))


FAMOUS_TANG = ["李白", "杜甫", "白居易", "王维", "李商隐", "杜牧", "孟浩然", "王昌龄",
                "刘禹锡", "岑参", "韦应物", "柳宗元", "韩愈", "元稹", "温庭筠"]
FAMOUS_SONG = ["苏轼", "辛弃疾", "陆游", "李清照", "欧阳修", "王安石", "柳永",
                "晏殊", "秦观", "周邦彦", "范仲淹", "杨万里", "范成大", "黄庭坚"]


if __name__ == "__main__":
    entries = load_all_poetry()
    stats = get_author_stats(entries)
    top20 = list(stats.items())[:20]
    print(f"\nTop 20 authors:")
    for name, count in top20:
        print(f"  {name}: {count}")
