import pandas as pd
import numpy as np


# =========================
# 0. 参数区：按你的实际列名改
# =========================
INPUT_FILE = "用户行为序列(2).xlsx"
OUTPUT_FILE = "scene_feature_table.xlsx"

USER_COL = "用户id"
TIME_COL = "行为时间戳"      # 形如: T-8/Tuesday/09:00
ACTION_COL = "action"      # 已转成英文；可能含 click&cart 这种
CONTENT_COL = "内容分类"    # main_meal / snack / other
ITEM_COL = "具体内容"       # 没有可改成 None

VALID_ACTIONS = {"search", "click", "cart", "collect", "order", "call"}


# =========================
# 1. 读取数据
# =========================
df = pd.read_excel(INPUT_FILE)

keep_cols = [USER_COL, TIME_COL, ACTION_COL, CONTENT_COL]
if ITEM_COL is not None and ITEM_COL in df.columns:
    keep_cols.append(ITEM_COL)

df = df[keep_cols].copy()
df = df.dropna(subset=[TIME_COL, ACTION_COL, CONTENT_COL])

for col in [USER_COL, TIME_COL, ACTION_COL, CONTENT_COL]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

if ITEM_COL is not None and ITEM_COL in df.columns:
    df[ITEM_COL] = df[ITEM_COL].astype(str).str.strip()


# =========================
# 2. 拆时间戳
# 假设格式: T-8/Tuesday/09:00
# =========================
split_cols = df[TIME_COL].str.split("/", expand=True)
df["t_day"] = split_cols[0]
df["weekday"] = split_cols[1]
df["hhmm"] = split_cols[2]

df["hour"] = pd.to_numeric(
    df["hhmm"].str.extract(r"(\d{1,2})", expand=False),
    errors="coerce"
)

weekend_set = {"星期六", "星期日"}
df["is_weekend"] = df["weekday"].isin(weekend_set).astype(int)


# =========================
# 3. 时间段映射
# 这是粗映射，后续你可自行调整
# =========================
conditions = [
    df["hour"].between(6, 9, inclusive="both"),
    df["hour"].between(10, 13, inclusive="both"),
    df["hour"].between(14, 17, inclusive="both"),
    df["hour"].between(18, 20, inclusive="both"),
    (df["hour"].between(21, 23, inclusive="both")) | (df["hour"].between(0, 5, inclusive="both"))
]
choices = ["breakfast", "lunch", "afternoon_tea", "dinner", "late_night"]

df["time_slot"] = np.select(conditions, choices, default="unknown")


# =========================
# 4. 构造 scene_id
# 你的“场景 id”
# =========================
df["scene_id"] = (
    df["weekday"].astype(str) + "_" +
    df[CONTENT_COL].astype(str) + "_" +
    df["time_slot"].astype(str)
)


# =========================
# 5. 拆复合动作
# click&cart -> click, cart
# =========================
df["action_list"] = df[ACTION_COL].str.split("&")
event_df = df.explode("action_list").copy()
event_df["action_atom"] = event_df["action_list"].astype(str).str.strip().str.lower()

event_df = event_df[event_df["action_atom"].isin(VALID_ACTIONS)].copy()


# =========================
# 6. scene-level 行为计数
# =========================
action_count = (
    event_df
    .pivot_table(
        index="scene_id",
        columns="action_atom",
        values=TIME_COL,
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

for col in ["search", "click", "cart", "collect", "order", "call"]:
    if col not in action_count.columns:
        action_count[col] = 0

action_count = action_count.rename(columns={
    "search": "n_search",
    "click": "n_click",
    "cart": "n_cart",
    "collect": "n_collect",
    "order": "n_order",
    "call": "n_call"
})


# =========================
# 6.1 scene 内按“单次访问”聚合的漏斗计数
# 用于观察 search/click -> commit -> call -> order 的推进
# =========================
event_df["scene_visit_id"] = (
    event_df[USER_COL].astype(str) + "|" +
    event_df[TIME_COL].astype(str) + "|" +
    event_df[CONTENT_COL].astype(str)
)

visit_key = "scene_visit_id"

visit_action = (
    event_df
    .pivot_table(
        index=["scene_id", visit_key],
        columns="action_atom",
        values=TIME_COL,
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

for col in ["search", "click", "cart", "collect", "order", "call"]:
    if col not in visit_action.columns:
        visit_action[col] = 0

visit_action["has_search_visit"] = (visit_action["search"] > 0).astype(int)
visit_action["has_click_visit"] = (visit_action["click"] > 0).astype(int)
visit_action["has_commit_visit"] = (
    (visit_action["cart"] > 0) | (visit_action["collect"] > 0)
).astype(int)
visit_action["has_call_visit"] = (visit_action["call"] > 0).astype(int)
visit_action["has_order_visit"] = (visit_action["order"] > 0).astype(int)
visit_action["has_explore_visit"] = (
    visit_action["has_search_visit"] | visit_action["has_click_visit"]
).astype(int)

visit_action["search_to_commit_visit"] = (
    visit_action["has_search_visit"] & visit_action["has_commit_visit"]
).astype(int)
visit_action["click_to_commit_visit"] = (
    visit_action["has_click_visit"] & visit_action["has_commit_visit"]
).astype(int)
visit_action["explore_to_commit_visit"] = (
    visit_action["has_explore_visit"] & visit_action["has_commit_visit"]
).astype(int)
visit_action["commit_to_call_visit"] = (
    visit_action["has_commit_visit"] & visit_action["has_call_visit"]
).astype(int)
visit_action["commit_to_order_visit"] = (
    visit_action["has_commit_visit"] & visit_action["has_order_visit"]
).astype(int)
visit_action["call_to_order_visit"] = (
    visit_action["has_call_visit"] & visit_action["has_order_visit"]
).astype(int)
visit_action["full_funnel_visit"] = (
    visit_action["has_explore_visit"] &
    visit_action["has_commit_visit"] &
    visit_action["has_call_visit"] &
    visit_action["has_order_visit"]
).astype(int)

scene_funnel = (
    visit_action.groupby("scene_id", as_index=False)
    .agg(
        n_scene_visits=(visit_key, "nunique"),
        n_search_visits=("has_search_visit", "sum"),
        n_click_visits=("has_click_visit", "sum"),
        n_explore_visits=("has_explore_visit", "sum"),
        n_commit_visits=("has_commit_visit", "sum"),
        n_call_visits=("has_call_visit", "sum"),
        n_order_visits=("has_order_visit", "sum"),
        has_search_to_commit=("search_to_commit_visit", "max"),
        has_click_to_commit=("click_to_commit_visit", "max"),
        has_explore_to_commit=("explore_to_commit_visit", "max"),
        has_commit_to_call=("commit_to_call_visit", "max"),
        has_commit_to_order=("commit_to_order_visit", "max"),
        has_call_to_order=("call_to_order_visit", "max"),
        has_full_funnel=("full_funnel_visit", "max"),
    )
)


# =========================
# 7. scene meta 信息
# =========================
agg_dict = {
    "weekday": ("weekday", "first"),
    "is_weekend": ("is_weekend", "first"),
    "content_type": (CONTENT_COL, "first"),
    "time_slot": ("time_slot", "first"),
    "n_rows": (TIME_COL, "count"),
    "n_users": (USER_COL, "nunique"),
    "n_days": ("t_day", "nunique"),
}

scene_meta = df.groupby("scene_id", as_index=False).agg(**agg_dict)

if ITEM_COL is not None and ITEM_COL in event_df.columns:
    item_df = (
        event_df.groupby("scene_id", as_index=False)
        .agg(n_unique_item=(ITEM_COL, "nunique"))
    )
else:
    item_df = pd.DataFrame({
        "scene_id": scene_meta["scene_id"],
        "n_unique_item": np.nan
    })


# =========================
# 8. 合并
# =========================
scene_feature = scene_meta.merge(action_count, on="scene_id", how="left")
scene_feature = scene_feature.merge(item_df, on="scene_id", how="left")
scene_feature = scene_feature.merge(scene_funnel, on="scene_id", how="left")


# =========================
# 9. 构造 feature
# count + dummy + ratio
# =========================
for col in ["n_search", "n_click", "n_cart", "n_collect", "n_order"]:
    scene_feature[col] = scene_feature[col].fillna(0)

for col in [
    "n_call",
    "n_scene_visits",
    "n_search_visits",
    "n_click_visits",
    "n_explore_visits",
    "n_commit_visits",
    "n_call_visits",
    "n_order_visits",
]:
    scene_feature[col] = scene_feature[col].fillna(0)

for col in [
    "has_search_to_commit",
    "has_click_to_commit",
    "has_explore_to_commit",
    "has_commit_to_call",
    "has_commit_to_order",
    "has_call_to_order",
    "has_full_funnel",
]:
    scene_feature[col] = scene_feature[col].fillna(0).astype(int)

scene_feature["n_actions"] = (
    scene_feature["n_search"] +
    scene_feature["n_click"] +
    scene_feature["n_cart"] +
    scene_feature["n_collect"] +
    scene_feature["n_order"] + 
    scene_feature["n_call"]
)

# 基础 dummy
scene_feature["has_search"] = (scene_feature["n_search"] > 0).astype(int)
scene_feature["has_click"] = (scene_feature["n_click"] > 0).astype(int)
scene_feature["has_cart"] = (scene_feature["n_cart"] > 0).astype(int)
scene_feature["has_collect"] = (scene_feature["n_collect"] > 0).astype(int)
scene_feature["has_order"] = (scene_feature["n_order"] > 0).astype(int)
scene_feature["has_call"] = (scene_feature["n_call"] > 0).astype(int)

scene_feature["has_commit"] = (
    (scene_feature["n_cart"] > 0) | (scene_feature["n_collect"] > 0)
).astype(int)

scene_feature["explore_to_commit"] = (
    ((scene_feature["n_search"] + scene_feature["n_click"]) > 0) &
    ((scene_feature["n_cart"] + scene_feature["n_collect"]) > 0)
).astype(int)

scene_feature["commit_to_order"] = (
    ((scene_feature["n_cart"] + scene_feature["n_collect"]) > 0) &
    (scene_feature["n_order"] > 0)
).astype(int)

scene_feature["has_commit_no_order"] = (
    ((scene_feature["n_cart"] + scene_feature["n_collect"]) > 0) &
    (scene_feature["n_order"] == 0)
).astype(int)

scene_feature["is_converted"] = (scene_feature["n_order"] > 0).astype(int)

# 强度指标
scene_feature["explore_count"] = scene_feature["n_search"] + scene_feature["n_click"]
scene_feature["commit_count"] = scene_feature["n_cart"] + scene_feature["n_collect"]

# 份额特征：聚类更适合用这些
denom = scene_feature["n_actions"].replace(0, np.nan)

scene_feature["search_share"] = scene_feature["n_search"] / denom
scene_feature["click_share"] = scene_feature["n_click"] / denom
scene_feature["cart_share"] = scene_feature["n_cart"] / denom
scene_feature["collect_share"] = scene_feature["n_collect"] / denom
scene_feature["order_share"] = scene_feature["n_order"] / denom

scene_feature["explore_share"] = scene_feature["explore_count"] / denom
scene_feature["commit_share"] = scene_feature["commit_count"] / denom

# 犹豫率：commit了但没完全转化的程度
scene_feature["hesitation_rate"] = np.where(
    scene_feature["commit_count"] > 0,
    (scene_feature["commit_count"] - scene_feature["n_order"]) / scene_feature["commit_count"],
    0
)

# 多内容比较
scene_feature["multi_item_compare_flag"] = np.where(
    scene_feature["n_unique_item"].fillna(0) >= 2,
    1,
    0
)

# 填补 ratio 空值
ratio_cols = [
    "search_share", "click_share", "cart_share", "collect_share", "order_share",
    "explore_share", "commit_share", "hesitation_rate"
]
scene_feature[ratio_cols] = scene_feature[ratio_cols].fillna(0)


# =========================
# 10. 排序 + 导出
# =========================
scene_feature = scene_feature.sort_values(
    by=["is_weekend", "weekday", "content_type", "time_slot"]
).reset_index(drop=True)

print("scene_feature shape:", scene_feature.shape)
print(scene_feature.head(20))

scene_feature.to_excel(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")
