from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MARKOV_FILE = OUTPUT_DIR / "state_transition_markov.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "state_logistic_models.xlsx"
SUMMARY_FILE = OUTPUT_DIR / "state_logistic_summary.md"


def load_event_samples() -> pd.DataFrame:
    df = pd.read_excel(MARKOV_FILE, sheet_name="01_event_state_samples")
    df = df.sort_values(["用户id", "event_idx"]).reset_index(drop=True)
    df = df[df["is_terminal_state"] == 0].copy()
    return df


def build_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    # Use previous states rather than future information. This is the direct
    # ML enhancement over the Markov baseline.
    work_df["prev_state_1"] = work_df.groupby("用户id")["state_name"].shift(1)
    work_df["prev_state_2"] = work_df.groupby("用户id")["state_name"].shift(2)
    work_df["prev_scene_1"] = work_df.groupby("用户id")["scene_key"].shift(1)
    work_df["prev_style_1"] = work_df.groupby("用户id")["style_name"].shift(1)

    state_rank_history = work_df.groupby("用户id")["state_rank"]
    work_df["prev_state_rank_1"] = state_rank_history.shift(1)
    work_df["prev_state_rank_2"] = state_rank_history.shift(2)

    for state_code, state_label in [
        ("S1_需求表达", "search"),
        ("S2_比较筛选", "compare"),
        ("S3_意向强化", "intent"),
        ("S4_转化完成", "convert"),
    ]:
        flag_col = f"is_{state_label}"
        work_df[flag_col] = (work_df["state_name"] == state_code).astype(int)
        work_df[f"recent_{state_label}_3"] = (
            work_df.groupby("用户id")[flag_col]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
            .fillna(0)
        )

    work_df["recent_progress_score"] = (
        work_df["recent_intent_3"] * 1.0
        + work_df["recent_convert_3"] * 1.5
        - work_df["recent_compare_3"] * 0.5
    )
    work_df["compare_intent_gap_3"] = work_df["recent_compare_3"] - work_df["recent_intent_3"]
    work_df["state_scene_key"] = work_df["state_name"].astype(str) + "|" + work_df["scene_key"].astype(str)
    work_df["state_style_key"] = work_df["state_name"].astype(str) + "|" + work_df["style_name"].fillna("未知").astype(str)

    # Keep only columns needed for modeling and interpretation.
    keep_cols = [
        "用户id",
        "event_idx",
        "scene_key",
        "day_type",
        "meal_period",
        "content_bucket",
        "style_name",
        "match_score",
        "state_name",
        "state_rank",
        "prev_state_1",
        "prev_state_2",
        "prev_scene_1",
        "prev_style_1",
        "prev_state_rank_1",
        "prev_state_rank_2",
        "recent_search_3",
        "recent_compare_3",
        "recent_intent_3",
        "recent_convert_3",
        "recent_progress_score",
        "compare_intent_gap_3",
        "state_scene_key",
        "state_style_key",
        "deepen_in_3",
        "convert_in_3",
        "next_state",
    ]
    return work_df[keep_cols].copy()


def fit_binary_logistic(
    data: pd.DataFrame,
    target_col: str,
    positive_label: str,
) -> dict:
    cat_features = [
        "state_name",
        "prev_state_1",
        "prev_state_2",
        "scene_key",
        "day_type",
        "meal_period",
        "content_bucket",
        "style_name",
        "prev_scene_1",
        "prev_style_1",
        "state_scene_key",
        "state_style_key",
    ]
    num_features = [
        "match_score",
        "state_rank",
        "prev_state_rank_1",
        "prev_state_rank_2",
        "recent_search_3",
        "recent_compare_3",
        "recent_intent_3",
        "recent_convert_3",
        "recent_progress_score",
        "compare_intent_gap_3",
    ]

    X = data[cat_features + num_features].copy()
    y = data[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "target": target_col,
        "positive_label": positive_label,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, pred_proba)),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        pred,
        labels=[1],
        zero_division=0,
    )
    metrics["precision_pos"] = float(precision[0])
    metrics["recall_pos"] = float(recall[0])
    metrics["f1_pos"] = float(f1[0])
    metrics["support_pos"] = int(support[0])

    report_df = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.reset_index()
    report_df = report_df.rename(columns={"index": "label"})

    cm = pd.DataFrame(
        confusion_matrix(y_test, pred, labels=[0, 1]),
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).reset_index().rename(columns={"index": "true_label"})

    result_df = X_test.copy()
    result_df["true_y"] = y_test.values
    result_df["pred_y"] = pred
    result_df["pred_prob"] = pred_proba

    coef_df = extract_coefficients(model, top_k=18)

    return {
        "metrics": pd.DataFrame([metrics]),
        "report": report_df,
        "confusion": cm,
        "result": result_df.sort_values("pred_prob", ascending=False).reset_index(drop=True),
        "coef": coef_df,
    }


def extract_coefficients(model: Pipeline, top_k: int = 15) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()
    coef = clf.coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)

    top_positive = coef_df.sort_values("coef", ascending=False).head(top_k).copy()
    top_positive["direction"] = "positive"
    top_negative = coef_df.sort_values("coef", ascending=True).head(top_k).copy()
    top_negative["direction"] = "negative"

    return pd.concat([top_positive, top_negative], ignore_index=True)


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
    model_data_df: pd.DataFrame,
    deepen_result: dict,
    convert_result: dict,
) -> None:
    lines = [
        "# Logistic 回归摘要",
        "",
        "## 为什么在 Markov 之后继续用 Logistic",
        "- Markov 结果显示 `S2_比较筛选` 是最主要停滞点，因此模型重点改为预测 `未来3步是否进入更深状态` 和 `未来3步是否转化`。",
        "- 在特征上加入 `当前状态 + 场景 + 风格 + 前两步状态 + 最近3步比较/意向次数`，用于吸收 Markov 无法表达的短期历史差异。",
        "- 使用 `class_weight='balanced'`，减少小样本正类被忽略的问题。",
        "",
        "## 样本概况",
        f"- 建模样本数：`{len(model_data_df)}`",
        f"- `deepen_in_3` 正类占比：`{model_data_df['deepen_in_3'].mean():.4f}`",
        f"- `convert_in_3` 正类占比：`{model_data_df['convert_in_3'].mean():.4f}`",
        "",
        "## deepen_in_3 模型指标",
        to_simple_markdown_table(
            deepen_result["metrics"][
                ["accuracy", "roc_auc", "precision_pos", "recall_pos", "f1_pos", "positive_rate_test"]
            ]
        ),
        "",
        "## convert_in_3 模型指标",
        to_simple_markdown_table(
            convert_result["metrics"][
                ["accuracy", "roc_auc", "precision_pos", "recall_pos", "f1_pos", "positive_rate_test"]
            ]
        ),
        "",
        "## deepen_in_3 主要正向/负向特征",
        to_simple_markdown_table(deepen_result["coef"][["direction", "feature", "coef"]].head(20)),
        "",
        "## convert_in_3 主要正向/负向特征",
        to_simple_markdown_table(convert_result["coef"][["direction", "feature", "coef"]].head(20)),
    ]

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_df = load_event_samples()
    model_data_df = build_modeling_table(event_df)

    deepen_result = fit_binary_logistic(
        model_data_df,
        target_col="deepen_in_3",
        positive_label="未来3步进入更深状态",
    )
    convert_result = fit_binary_logistic(
        model_data_df,
        target_col="convert_in_3",
        positive_label="未来3步转化",
    )

    metrics_df = pd.concat([deepen_result["metrics"], convert_result["metrics"]], ignore_index=True)

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        model_data_df.to_excel(writer, sheet_name="01_model_samples", index=False)
        metrics_df.to_excel(writer, sheet_name="02_model_metrics", index=False)

        deepen_result["report"].to_excel(writer, sheet_name="03_deepen_report", index=False)
        deepen_result["confusion"].to_excel(writer, sheet_name="04_deepen_confusion", index=False)
        deepen_result["coef"].to_excel(writer, sheet_name="05_deepen_coef", index=False)
        deepen_result["result"].to_excel(writer, sheet_name="06_deepen_predictions", index=False)

        convert_result["report"].to_excel(writer, sheet_name="07_convert_report", index=False)
        convert_result["confusion"].to_excel(writer, sheet_name="08_convert_confusion", index=False)
        convert_result["coef"].to_excel(writer, sheet_name="09_convert_coef", index=False)
        convert_result["result"].to_excel(writer, sheet_name="10_convert_predictions", index=False)

    write_summary_markdown(model_data_df, deepen_result, convert_result)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {SUMMARY_FILE}")
    print("\nModel metrics:")
    print(metrics_df.to_string(index=False))
    print("\nTop convert_in_3 coefficients:")
    print(convert_result["coef"][["direction", "feature", "coef"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
