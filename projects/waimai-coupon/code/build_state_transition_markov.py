from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

BEHAVIOR_FILE = PROJECT_DIR / "用户行为序列(3).xlsx"
SCENE_STYLE_FILE = PROJECT_DIR / "场景风格融合" / "outputs" / "scene_style_master.xlsx"

OUTPUT_FILE = OUTPUT_DIR / "state_transition_markov.xlsx"
SUMMARY_FILE = OUTPUT_DIR / "state_transition_summary.md"

WEEKEND_SET = {"星期六", "星期日"}

ACTION_NORMALIZE_MAP = {
    "搜索": "search",
    "search": "search",
    "点击店铺": "click",
    "click": "click",
    "浏览详情页": "click",
    "加入购物车": "cart",
    "cart": "cart",
    "收藏": "collect",
    "collect": "collect",
    "下单": "order",
    "order": "order",
    "点击并收藏": "click&collect",
    "点击并加购": "click&cart",
    "cart&click": "click&cart",
    "click&cart": "click&cart",
    "click&collect": "click&collect",
    "click&call&collect": "click&collect",
}

ACTION_ORDER = {
    "search": 1,
    "click": 2,
    "cart": 3,
    "collect": 3,
    "order": 4,
}

STATE_INFO = {
    "search": ("S1_需求表达", 1),
    "click": ("S2_比较筛选", 2),
    "cart": ("S3_意向强化", 3),
    "collect": ("S3_意向强化", 3),
    "order": ("S4_转化完成", 4),
}

CONTENT_BUCKET_RULES = [
    ("泛浏览入口", r"^外卖$|^美团外卖$|^美食$|^团购$|水果鲜花|万象城"),
    ("饮品甜点", r"奶茶|咖啡|茶饮|果汁|鲜果|柠檬|芝士|蛋糕|甜点|冰淇淋|冰咖|轻乳茶|水果捞|纯茶|凉茶|1点点|煮茶|黑糖"),
    ("正餐快餐", r"快餐|简餐|正餐|卤肉饭|拌饭|牛堡|汉堡|麦当劳|麦满分|披萨|牛排|鸡腿|套餐|米饭|牛腩|饭|面|粉|粥|火锅|料理|小吃|麻辣烫|牛杂"),
]


def meal_period_from_hour(hour: float) -> str:
    if pd.isna(hour):
        return "unknown"
    hour = int(hour)
    if 6 <= hour <= 9:
        return "早餐"
    if 10 <= hour <= 13:
        return "午餐"
    if 14 <= hour <= 17:
        return "下午茶"
    if 18 <= hour <= 20:
        return "晚餐"
    return "夜宵"


def map_content_bucket(text: str) -> str:
    text = str(text)
    for bucket, pattern in CONTENT_BUCKET_RULES:
        if re.search(pattern, text):
            return bucket
    return "其他"


def read_scene_style_mapping() -> pd.DataFrame:
    if not SCENE_STYLE_FILE.exists():
        return pd.DataFrame(columns=["scene_key", "style_name", "match_score", "strategy_hint"])
    return pd.read_excel(
        SCENE_STYLE_FILE,
        sheet_name="05_style_match_best",
        usecols=["scene_key", "style_name", "match_score", "strategy_hint"],
    )


def build_event_state_table() -> pd.DataFrame:
    df = pd.read_excel(BEHAVIOR_FILE).copy()
    df["row_id"] = np.arange(len(df))

    split_cols = df["行为时间戳"].astype(str).str.split("/", expand=True)
    df["rel_day"] = split_cols[0]
    df["weekday"] = split_cols[1]
    df["hhmm"] = split_cols[2]
    rel_day_num = pd.to_numeric(df["rel_day"].astype(str).str.extract(r"T-(\d+)", expand=False), errors="coerce")
    df["time_rank_day"] = -rel_day_num
    df["hour"] = pd.to_numeric(df["hhmm"].str.extract(r"(\d{1,2})", expand=False), errors="coerce")
    df["minute"] = pd.to_numeric(df["hhmm"].str.extract(r":(\d{2})", expand=False), errors="coerce")
    df["day_type"] = np.where(df["weekday"].isin(WEEKEND_SET), "周末", "工作日")
    df["meal_period"] = df["hour"].map(meal_period_from_hour)
    df["content_bucket"] = df["具体内容"].map(map_content_bucket)
    df["scene_key"] = (
        df["day_type"].astype(str)
        + "_"
        + df["meal_period"].astype(str)
        + "_"
        + df["content_bucket"].astype(str)
    )

    df["behavior_std"] = df["行为类型"].astype(str).map(ACTION_NORMALIZE_MAP).fillna(df["行为类型"].astype(str))
    df["action_list"] = df["behavior_std"].astype(str).str.split("&")
    event_df = df.explode("action_list").copy()
    event_df["action_atom"] = event_df["action_list"].astype(str).str.strip().str.lower()
    event_df = event_df[event_df["action_atom"].isin(STATE_INFO.keys())].copy()

    event_df["action_order"] = event_df["action_atom"].map(ACTION_ORDER)
    event_df["state_name"] = event_df["action_atom"].map(lambda x: STATE_INFO[x][0])
    event_df["state_rank"] = event_df["action_atom"].map(lambda x: STATE_INFO[x][1])

    style_df = read_scene_style_mapping()
    event_df = event_df.merge(style_df, on="scene_key", how="left")

    event_df = event_df.sort_values(
        ["用户id", "time_rank_day", "hour", "minute", "session_id", "row_id", "action_order"],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)

    event_df["event_idx"] = event_df.groupby("用户id").cumcount() + 1
    event_df["next_state"] = event_df.groupby("用户id")["state_name"].shift(-1)
    event_df["next_state_rank"] = event_df.groupby("用户id")["state_rank"].shift(-1)
    event_df["next_scene_key"] = event_df.groupby("用户id")["scene_key"].shift(-1)

    max_horizon = 3
    future_rank_cols = []
    future_state_cols = []
    for step in range(1, max_horizon + 1):
        rank_col = f"future_state_rank_t{step}"
        state_col = f"future_state_t{step}"
        event_df[rank_col] = event_df.groupby("用户id")["state_rank"].shift(-step)
        event_df[state_col] = event_df.groupby("用户id")["state_name"].shift(-step)
        future_rank_cols.append(rank_col)
        future_state_cols.append(state_col)

    event_df["max_future_rank_3"] = event_df[future_rank_cols].max(axis=1, skipna=True)
    event_df["deepen_in_3"] = (
        event_df["max_future_rank_3"].fillna(event_df["state_rank"]) > event_df["state_rank"]
    ).astype(int)
    event_df["convert_in_3"] = (
        event_df[future_state_cols].eq("S4_转化完成").any(axis=1)
    ).astype(int)

    event_df["is_terminal_state"] = event_df["next_state"].isna().astype(int)
    return event_df


def build_transition_tables(event_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_df = event_df[event_df["next_state"].notna()].copy()

    global_counts = (
        transition_df.groupby(["state_name", "next_state"], as_index=False)
        .agg(transition_cnt=("用户id", "size"))
        .rename(columns={"state_name": "from_state", "next_state": "to_state"})
    )
    total_by_from = global_counts.groupby("from_state")["transition_cnt"].transform("sum")
    global_counts["transition_prob"] = global_counts["transition_cnt"] / total_by_from

    scene_counts = (
        transition_df.groupby(["scene_key", "style_name", "state_name", "next_state"], as_index=False)
        .agg(transition_cnt=("用户id", "size"))
        .rename(columns={"state_name": "from_state", "next_state": "to_state"})
    )
    scene_total = scene_counts.groupby(["scene_key", "from_state"])["transition_cnt"].transform("sum")
    scene_counts["transition_prob"] = scene_counts["transition_cnt"] / scene_total
    return global_counts, scene_counts


def build_state_scene_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        event_df.groupby(["scene_key", "style_name", "state_name"], as_index=False)
        .agg(
            event_cnt=("用户id", "size"),
            user_cnt=("用户id", "nunique"),
            deepen_in_3_rate=("deepen_in_3", "mean"),
            convert_in_3_rate=("convert_in_3", "mean"),
        )
        .sort_values(["scene_key", "state_name"])
        .reset_index(drop=True)
    )
    return summary_df


def build_scene_path_profile(event_df: pd.DataFrame) -> pd.DataFrame:
    profile_df = (
        event_df.groupby(["scene_key", "style_name"], as_index=False)
        .agg(
            event_cnt=("用户id", "size"),
            user_cnt=("用户id", "nunique"),
            search_share=("state_name", lambda s: float((s == "S1_需求表达").mean())),
            compare_share=("state_name", lambda s: float((s == "S2_比较筛选").mean())),
            intent_share=("state_name", lambda s: float((s == "S3_意向强化").mean())),
            convert_share=("state_name", lambda s: float((s == "S4_转化完成").mean())),
            deepen_in_3_rate=("deepen_in_3", "mean"),
            convert_in_3_rate=("convert_in_3", "mean"),
        )
        .sort_values(["event_cnt", "scene_key"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return profile_df


def to_simple_markdown_table(df: pd.DataFrame) -> str:
    show_df = df.copy()
    for col in show_df.columns:
        if pd.api.types.is_float_dtype(show_df[col]):
            show_df[col] = show_df[col].map(lambda x: f"{x:.4f}")

    headers = [str(col) for col in show_df.columns]
    rows = [[str(v) for v in row] for row in show_df.to_numpy()]
    widths = [len(h) for h in headers]

    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def write_summary_markdown(
    global_counts: pd.DataFrame,
    scene_summary: pd.DataFrame,
    scene_profile: pd.DataFrame,
) -> None:
    top_global = global_counts.sort_values("transition_prob", ascending=False).head(12).copy()
    top_scenes = scene_profile.sort_values("convert_in_3_rate", ascending=False).head(10).copy()
    top_scene_states = scene_summary.sort_values("convert_in_3_rate", ascending=False).head(12).copy()

    lines = [
        "# 状态转移摘要",
        "",
        "## 全局高概率转移",
        to_simple_markdown_table(top_global),
        "",
        "## 高短期转化场景",
        to_simple_markdown_table(
            top_scenes[
                [
                    "scene_key",
                    "style_name",
                    "event_cnt",
                    "compare_share",
                    "intent_share",
                    "convert_share",
                    "deepen_in_3_rate",
                    "convert_in_3_rate",
                ]
            ]
        ),
        "",
        "## 场景-状态高短期转化组合",
        to_simple_markdown_table(
            top_scene_states[
                ["scene_key", "style_name", "state_name", "event_cnt", "deepen_in_3_rate", "convert_in_3_rate"]
            ]
        ),
    ]

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_df = build_event_state_table()
    global_counts, scene_counts = build_transition_tables(event_df)
    scene_summary = build_state_scene_summary(event_df)
    scene_profile = build_scene_path_profile(event_df)

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        event_df.to_excel(writer, sheet_name="01_event_state_samples", index=False)
        global_counts.to_excel(writer, sheet_name="02_global_transition", index=False)
        scene_counts.to_excel(writer, sheet_name="03_scene_transition", index=False)
        scene_summary.to_excel(writer, sheet_name="04_scene_state_summary", index=False)
        scene_profile.to_excel(writer, sheet_name="05_scene_path_profile", index=False)

    write_summary_markdown(global_counts, scene_summary, scene_profile)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print("\nTop global transitions:")
    print(global_counts.sort_values("transition_prob", ascending=False).head(10).to_string(index=False))
    print("\nTop scene-state combinations by convert_in_3_rate:")
    print(
        scene_summary.sort_values("convert_in_3_rate", ascending=False)
        .head(10)[["scene_key", "style_name", "state_name", "event_cnt", "convert_in_3_rate"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
