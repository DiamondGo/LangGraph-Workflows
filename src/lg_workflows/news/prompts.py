"""Output-language dependent prompts and labels.

Only "zh" is implemented. To add English, add an "en" entry to PROMPTS and LABELS;
nothing else needs to change.
"""

from typing import Literal

Lang = Literal["zh", "en"]

# Classification only emits topic keys, so it is language-neutral.
CLASSIFY = """Classify each news headline into exactly one topic:
- us_politics: US government, elections, Congress, courts, US political parties and figures
- china_politics: politics, government, policy, diplomacy, military or human rights involving China, Hong Kong or Taiwan
- ai: artificial intelligence, machine learning, LLMs, AI companies, products, research or regulation
- other: anything else
Return one label per headline index.

Headlines:
{headlines}"""

PROMPTS: dict[str, dict[str, str]] = {
    "zh": {
        "summarize": """你是新闻编辑。根据下面同一事件的多篇报道，用简体中文写：
1. title：一句话中文标题（不超过 30 字），客观中立。
2. summary：2-4 句事实摘要，说明发生了什么、涉及谁、为什么重要；不同来源说法有分歧时要指出。
3. comments_summary：{comments_instruction}
只依据提供的材料，不要编造。

主题：{topic}

报道：
{articles}
{comments}""",
        "comments_instruction_on": "根据 Hacker News 评论，用 2-4 句总结主要观点、主要分歧和有价值的补充信息，以及整体情绪。",
        "comments_instruction_off": "没有评论，返回空字符串。",
        "daily": """你是新闻编辑。下面是今天（{date}）每小时简报中的事件摘要，可能有同一事件的多次更新。
请用简体中文写一份日报（Markdown）：
- 按主题分节：{topics}
- 每节合并重复事件，挑出最重要的 5-8 件，每件写 2-3 句，说明进展和意义；有评论观点的简要带上。
- 最后写一段“今日要点”，3-5 条，跨主题总结最值得关注的事。
- 客观中立，只依据材料，不要编造，不要输出原始链接。

事件摘要：
{events}""",
    },
}

LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "hourly_title": "新闻简报",
        "daily_title": "新闻日报",
        "sources": "来源",
        "heat": "热度",
        "freshness": "最新报道",
        "minutes_ago": "{n} 分钟前",
        "hours_ago": "{n} 小时前",
        "comments": "评论观点",
        "comment_count": "Hacker News {n} 条评论",
        "links": "链接",
        "no_events": "本时段没有新的事件。",
        "errors": "抓取失败的来源",
    },
}


def check_lang(lang: str) -> None:
    if lang not in PROMPTS:
        raise NotImplementedError(f"output language {lang!r} is not implemented yet (available: {list(PROMPTS)})")
