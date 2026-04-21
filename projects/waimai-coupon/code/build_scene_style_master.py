from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

BEHAVIOR_FILE = PROJECT_DIR / "用户行为序列(3).xlsx"
ORDER_FILE = PROJECT_DIR / "神券订单数据样例（仅）外卖 (1)(1).xlsx"
STYLE_FILE = PROJECT_DIR / "cluster_profiles_summary.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "scene_style_master.xlsx"
SUMMARY_FILE = OUTPUT_DIR / "scene_style_summary.md"


WEEKEND_SET = {"星期六", "星期日"}
BEHAVIOR_ACTION_MAP = {
    "搜索": "search",
    "search": "search",
    "点击店铺": "click",
    "click": "click",
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

BEHAVIOR_BUCKET_RULES = [
    ("泛浏览入口", r"^外卖$|^美团外卖$|^美食$|^团购$|水果鲜花|万象城"),
    ("饮品甜点", r"奶茶|咖啡|茶饮|果汁|鲜果|柠檬|芝士|蛋糕|甜点|冰淇淋|冰咖|轻乳茶|水果捞|纯茶|凉茶|1点点|煮茶|黑糖"),
    ("正餐快餐", r"快餐|简餐|正餐|卤肉饭|拌饭|牛堡|汉堡|麦当劳|麦满分|披萨|牛排|鸡腿|套餐|米饭|牛腩|饭|面|粉|粥|火锅|料理|小吃|麻辣烫|牛杂"),
]

ORDER_BUCKET_MAP = {
    "快餐简餐": "正餐快餐",
    "中式正餐": "正餐快餐",
    "全球美食": "正餐快餐",
    "小吃": "正餐快餐",
    "火锅": "正餐快餐",
    "奶茶": "饮品甜点",
    "咖啡": "饮品甜点",
    "生日蛋糕": "饮品甜点",
    "西式点心": "饮品甜点",
    "冰凉甜点": "饮品甜点",
    "纯茶/凉茶": "饮品甜点",
    "其他甜点": "饮品甜点",
    "其他饮品": "饮品甜点",
    "水果捞": "饮品甜点",
}

SPEND_SCORE_MAP = {
    "低客单价": 0.25,
    "中客单价": 0.50,
    "高客单价": 0.75,
    "超高客单价": 1.00,
}
MEMBER_SCORE_MAP = {"L1": 0.15, "L2": 0.30, "L3": 0.45, "L4": 0.65, "L5": 0.82, "L6": 1.00}
LIFECYCLE_SCORE_MAP = {"新客": 0.25, "老客": 0.80, "流失召回": 0.45}


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


def minmax_scale(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - min_value) / (max_value - min_value)


def map_behavior_bucket(text: str) -> str:
    text = str(text)
    for bucket, pattern in BEHAVIOR_BUCKET_RULES:
        if re.search(pattern, text):
            return bucket
    return "其他"


def normalize_behavior_actions(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df["行为类型_std"] = work_df["行为类型"].astype(str).map(BEHAVIOR_ACTION_MAP).fillna(
        work_df["行为类型"].astype(str)
    )
    work_df["action_list"] = work_df["行为类型_std"].astype(str).str.split("&")
    event_df = work_df.explode("action_list").copy()
    event_df["action_atom"] = event_df["action_list"].astype(str).str.strip().str.lower()
    return event_df


def build_behavior_scene_profile() -> pd.DataFrame:
    df = pd.read_excel(BEHAVIOR_FILE)
    split_cols = df["行为时间戳"].astype(str).str.split("/", expand=True)
    df["rel_day"] = split_cols[0]
    df["weekday"] = split_cols[1]
    df["hhmm"] = split_cols[2]
    df["hour"] = pd.to_numeric(df["hhmm"].str.extract(r"(\d{1,2})", expand=False), errors="coerce")
    df["day_type"] = np.where(df["weekday"].isin(WEEKEND_SET), "周末", "工作日")
    df["meal_period"] = df["hour"].map(meal_period_from_hour)
    df["content_bucket"] = df["具体内容"].map(map_behavior_bucket)

    event_df = normalize_behavior_actions(df)
    event_df["scene_key"] = (
        event_df["day_type"].astype(str)
        + "_"
        + event_df["meal_period"].astype(str)
        + "_"
        + event_df["content_bucket"].astype(str)
    )

    scene_df = (
        event_df.groupby(["scene_key", "day_type", "weekday", "meal_period", "content_bucket"], as_index=False)
        .agg(
            behavior_rows=("用户id", "size"),
            behavior_users=("用户id", "nunique"),
            rel_days=("rel_day", "nunique"),
            unique_content_cnt=("具体内容", "nunique"),
            search_cnt=("action_atom", lambda s: int((s == "search").sum())),
            click_cnt=("action_atom", lambda s: int((s == "click").sum())),
            intent_cnt=("action_atom", lambda s: int(s.isin(["cart", "collect"]).sum())),
            order_cnt=("action_atom", lambda s: int((s == "order").sum())),
        )
        .sort_values(["behavior_rows", "scene_key"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_actions = scene_df[["search_cnt", "click_cnt", "intent_cnt", "order_cnt"]].sum(axis=1).replace(0, np.nan)
    scene_df["search_share"] = scene_df["search_cnt"] / total_actions
    scene_df["click_share"] = scene_df["click_cnt"] / total_actions
    scene_df["intent_share"] = scene_df["intent_cnt"] / total_actions
    scene_df["order_share"] = scene_df["order_cnt"] / total_actions
    scene_df["content_per_user"] = scene_df["unique_content_cnt"] / scene_df["behavior_users"].replace(0, np.nan)
    scene_df["conversion_signal"] = scene_df["order_cnt"] / scene_df["behavior_rows"].replace(0, np.nan)
    scene_df["exploration_proxy_raw"] = (
        scene_df["search_share"].fillna(0) * 0.40
        + scene_df["click_share"].fillna(0) * 0.25
        + minmax_scale(scene_df["content_per_user"].fillna(0)) * 0.35
    )
    return scene_df


def build_order_profile_scene_env() -> pd.DataFrame:
    order_df = pd.read_excel(ORDER_FILE, sheet_name="交易订单表")
    profile_df = pd.read_excel(ORDER_FILE, sheet_name="用户画像表")

    order_df["hour"] = pd.to_datetime(order_df["下单时间"].astype(str), format="%H:%M:%S", errors="coerce").dt.hour
    order_df["meal_period"] = order_df["hour"].map(meal_period_from_hour)
    order_df["content_bucket"] = order_df["POI分类"].map(ORDER_BUCKET_MAP).fillna("其他")
    order_df["free_coupon_flag"] = order_df["1-免费神券，2-付费神券"].eq(1).astype(float)
    order_df["inflate_coupon_flag"] = order_df["1-非膨胀券，2-膨胀券"].eq(2).astype(float)
    order_df["subsidy_rate"] = np.where(
        order_df["订单金额"].gt(0),
        order_df["美补金额"] / order_df["订单金额"],
        np.nan,
    )

    profile_df["spend_score"] = profile_df["历史实付分层（365天）"].map(SPEND_SCORE_MAP)
    profile_df["member_score"] = profile_df["会员状态"].map(MEMBER_SCORE_MAP)
    profile_df["lifecycle_score"] = profile_df["用户生命周期阶段"].map(LIFECYCLE_SCORE_MAP)
    profile_df["is_old_user"] = profile_df["用户生命周期阶段"].eq("老客").astype(float)
    profile_df["is_new_user"] = profile_df["用户生命周期阶段"].eq("新客").astype(float)
    profile_df["is_recall_user"] = profile_df["用户生命周期阶段"].eq("流失召回").astype(float)
    profile_df["is_high_member"] = profile_df["会员状态"].isin(["L4", "L5", "L6"]).astype(float)
    profile_df["is_high_spend"] = profile_df["历史实付分层（365天）"].isin(["高客单价", "超高客单价"]).astype(float)
    profile_df["is_low_spend"] = profile_df["历史实付分层（365天）"].eq("低客单价").astype(float)

    merged_df = order_df.merge(profile_df, on="user_id", how="left")
    scene_env = (
        merged_df.groupby(["meal_period", "content_bucket"], as_index=False)
        .agg(
            order_scene_cnt=("order_id", "size"),
            order_users=("user_id", "nunique"),
            avg_order_amt=("订单金额", "mean"),
            avg_subsidy_amt=("美补金额", "mean"),
            avg_subsidy_rate=("subsidy_rate", "mean"),
            free_coupon_rate=("free_coupon_flag", "mean"),
            inflate_coupon_rate=("inflate_coupon_flag", "mean"),
            avg_age=("年龄", "mean"),
            old_user_rate=("is_old_user", "mean"),
            new_user_rate=("is_new_user", "mean"),
            recall_user_rate=("is_recall_user", "mean"),
            high_member_rate=("is_high_member", "mean"),
            high_spend_rate=("is_high_spend", "mean"),
            low_spend_rate=("is_low_spend", "mean"),
            spend_score_mean=("spend_score", "mean"),
            member_score_mean=("member_score", "mean"),
            lifecycle_score_mean=("lifecycle_score", "mean"),
        )
        .sort_values(["order_scene_cnt", "meal_period", "content_bucket"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return scene_env


def build_style_match_tables(scene_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    style_df = pd.read_excel(STYLE_FILE).rename(
        columns={
            "类型": "style_name",
            "Cluster_ID": "cluster_id",
            "用户数量": "style_user_cnt",
            "用户占比": "style_user_share",
            "平均客单价": "style_avg_order_amt",
            "整体美补率": "style_subsidy_rate",
            "免费神券占比": "style_free_coupon_rate",
            "用户价值度分": "style_value_score",
            "价格敏感度分": "style_price_score",
            "探索尝鲜度分": "style_explore_score",
        }
    )

    work_df = scene_master.copy()
    work_df["scene_value_score"] = (
        work_df["avg_order_amt_norm"].fillna(0.5) * 0.35
        + work_df["spend_score_mean"].fillna(0.5) * 0.30
        + work_df["high_member_rate"].fillna(0.5) * 0.20
        + work_df["high_spend_rate"].fillna(0.5) * 0.15
    )
    work_df["scene_price_score"] = (
        work_df["avg_subsidy_rate_norm"].fillna(0.5) * 0.45
        + work_df["free_coupon_rate"].fillna(0.5) * 0.30
        + work_df["low_spend_rate"].fillna(0.5) * 0.25
    )
    work_df["scene_explore_score"] = (
        work_df["exploration_proxy"].fillna(0.5) * 0.70
        + work_df["click_share"].fillna(0) * 0.15
        + work_df["search_share"].fillna(0) * 0.15
    )

    combined_order_values = pd.concat(
        [work_df["avg_order_amt"].fillna(work_df["avg_order_amt"].median()), style_df["style_avg_order_amt"]],
        ignore_index=True,
    )
    combined_subsidy_values = pd.concat(
        [work_df["avg_subsidy_rate"].fillna(work_df["avg_subsidy_rate"].median()), style_df["style_subsidy_rate"]],
        ignore_index=True,
    )
    order_min, order_max = combined_order_values.min(), combined_order_values.max()
    subsidy_min, subsidy_max = combined_subsidy_values.min(), combined_subsidy_values.max()

    def norm_with_bounds(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
        if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
            return pd.Series(np.full(len(series), 0.5), index=series.index)
        return (series - min_value) / (max_value - min_value)

    work_df["avg_order_amt_norm_cross"] = norm_with_bounds(
        work_df["avg_order_amt"].fillna(work_df["avg_order_amt"].median()),
        order_min,
        order_max,
    )
    style_df["style_avg_order_amt_norm"] = norm_with_bounds(style_df["style_avg_order_amt"], order_min, order_max)
    style_df["style_subsidy_rate_norm"] = norm_with_bounds(style_df["style_subsidy_rate"], subsidy_min, subsidy_max)

    candidates = work_df.assign(_join_key=1).merge(style_df.assign(_join_key=1), on="_join_key", how="inner").drop(
        columns="_join_key"
    )

    candidates["value_similarity"] = 1 - (candidates["scene_value_score"] - candidates["style_value_score"]).abs()
    candidates["price_similarity"] = 1 - (candidates["scene_price_score"] - candidates["style_price_score"]).abs()
    candidates["explore_similarity"] = 1 - (candidates["scene_explore_score"] - candidates["style_explore_score"]).abs()
    candidates["order_amt_similarity"] = 1 - (
        candidates["avg_order_amt_norm_cross"] - candidates["style_avg_order_amt_norm"]
    ).abs()
    candidates["subsidy_similarity"] = 1 - (
        candidates["avg_subsidy_rate_norm"].fillna(0.5) - candidates["style_subsidy_rate_norm"]
    ).abs()

    similarity_cols = [
        "value_similarity",
        "price_similarity",
        "explore_similarity",
        "order_amt_similarity",
        "subsidy_similarity",
    ]
    candidates[similarity_cols] = candidates[similarity_cols].clip(lower=0, upper=1)
    candidates["match_score"] = (
        candidates["value_similarity"] * 0.30
        + candidates["price_similarity"] * 0.20
        + candidates["explore_similarity"] * 0.30
        + candidates["order_amt_similarity"] * 0.10
        + candidates["subsidy_similarity"] * 0.10
    )

    candidates["match_reason"] = candidates.apply(build_match_reason, axis=1)
    candidates = candidates.sort_values(["scene_key", "match_score"], ascending=[True, False]).reset_index(drop=True)

    best_match = candidates.groupby("scene_key", as_index=False).head(1).copy()
    best_match["strategy_hint"] = best_match.apply(build_strategy_hint, axis=1)
    return candidates, best_match


def build_match_reason(row: pd.Series) -> str:
    reasons = []
    if row["value_similarity"] >= 0.8:
        reasons.append("价值层级接近")
    if row["explore_similarity"] >= 0.8:
        reasons.append("探索倾向接近")
    if row["price_similarity"] >= 0.8 or row["subsidy_similarity"] >= 0.8:
        reasons.append("补贴敏感特征接近")
    if row["order_amt_similarity"] >= 0.8:
        reasons.append("客单价水平接近")
    if not reasons:
        reasons.append("三类代理分数整体最接近")
    return "；".join(reasons)


def build_strategy_hint(row: pd.Series) -> str:
    if row["style_name"] == "低价值高尝鲜型":
        return "更适合尝鲜券、小额刺激券或低门槛引导券。"
    if row["style_name"] == "高价值低尝鲜型":
        return "更适合弱刺激复购券或高效率成交券，避免过强补贴。"
    if row["style_name"] == "核心优质型":
        return "更适合精细化权益、会员感知型补贴或轻触达策略。"
    return "更适合价格友好型补贴，但需控制补贴深度以防低效消耗。"


def build_scene_master(
    behavior_scene_df: pd.DataFrame,
    order_scene_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = behavior_scene_df.merge(
        order_scene_df,
        on=["meal_period", "content_bucket"],
        how="left",
    )

    merged_df["avg_order_amt_norm"] = minmax_scale(merged_df["avg_order_amt"].fillna(merged_df["avg_order_amt"].median()))
    merged_df["avg_subsidy_rate_norm"] = minmax_scale(
        merged_df["avg_subsidy_rate"].fillna(merged_df["avg_subsidy_rate"].median())
    )
    merged_df["exploration_proxy"] = minmax_scale(merged_df["exploration_proxy_raw"].fillna(0))

    merged_df["behavior_scene_summary"] = merged_df.apply(build_behavior_summary, axis=1)
    merged_df["order_env_summary"] = merged_df.apply(build_order_summary, axis=1)
    return merged_df


def build_behavior_summary(row: pd.Series) -> str:
    tags = []
    if row["search_share"] >= 0.25:
        tags.append("搜索驱动较强")
    if row["click_share"] >= 0.40:
        tags.append("比较浏览较强")
    if row["intent_share"] >= 0.25:
        tags.append("加购承诺信号较强")
    if row["order_share"] >= 0.15:
        tags.append("即时转化信号较强")
    if row["content_per_user"] >= 2:
        tags.append("内容探索度较高")
    if not tags:
        tags.append("行为特征相对平缓")
    return "；".join(tags)


def build_order_summary(row: pd.Series) -> str:
    if pd.isna(row["avg_order_amt"]):
        return "订单环境缺失，仅能基于行为侧做弱匹配。"

    tags = []
    if row["avg_order_amt"] >= 45:
        tags.append("客单价偏高")
    elif row["avg_order_amt"] <= 35:
        tags.append("客单价偏低")
    else:
        tags.append("客单价中等")

    if row["avg_subsidy_rate"] >= 0.25:
        tags.append("补贴依赖偏强")
    elif row["avg_subsidy_rate"] <= 0.18:
        tags.append("补贴依赖偏弱")

    if row["free_coupon_rate"] >= 0.55:
        tags.append("免费券占比较高")
    if row["high_spend_rate"] >= 0.35:
        tags.append("高消费用户占比偏高")
    if row["new_user_rate"] >= 0.35:
        tags.append("新客占比偏高")
    return "；".join(tags)


def to_simple_markdown_table(df: pd.DataFrame) -> str:
    show_df = df.copy()
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


def write_summary_markdown(best_match_df: pd.DataFrame) -> None:
    lines = [
        "# 场景风格匹配摘要",
        "",
        "下表展示了部分场景的最佳风格匹配结果。",
        "",
    ]
    show_cols = [
        "scene_key",
        "behavior_scene_summary",
        "order_env_summary",
        "style_name",
        "match_score",
        "match_reason",
        "strategy_hint",
    ]
    preview_df = best_match_df.sort_values("match_score", ascending=False).head(12)[show_cols].copy()
    preview_df["match_score"] = preview_df["match_score"].map(lambda x: f"{x:.4f}")
    lines.append(to_simple_markdown_table(preview_df))
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    behavior_scene_df = build_behavior_scene_profile()
    order_scene_df = build_order_profile_scene_env()
    scene_master_df = build_scene_master(behavior_scene_df, order_scene_df)
    style_candidates_df, style_best_df = build_style_match_tables(scene_master_df)

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        behavior_scene_df.to_excel(writer, sheet_name="01_behavior_scene_profile", index=False)
        order_scene_df.to_excel(writer, sheet_name="02_order_profile_scene_env", index=False)
        scene_master_df.to_excel(writer, sheet_name="03_scene_master_merged", index=False)
        style_candidates_df.to_excel(writer, sheet_name="04_style_match_candidates", index=False)
        style_best_df.to_excel(writer, sheet_name="05_style_match_best", index=False)

    write_summary_markdown(style_best_df)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print("\nTop 8 matched scenes:")
    print(
        style_best_df[
            [
                "scene_key",
                "style_name",
                "match_score",
                "behavior_scene_summary",
                "order_env_summary",
            ]
        ]
        .sort_values("match_score", ascending=False)
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
