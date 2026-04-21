from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
ASSET_DIR = OUTPUT_DIR / "report_assets"
INPUT_FILE = OUTPUT_DIR / "state_logistic_models.xlsx"

STATE_COLORS = {
    "S1": "#DCE5EE",
    "S2": "#AEBED0",
    "S3": "#7F97B1",
    "S4": "#5F7894",
}


def roc_points(y_true: Iterable[int], y_score: Iterable[float]) -> tuple[list[tuple[float, float]], float]:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    total_pos = sum(y_true)
    total_neg = len(list(y_true)) - total_pos
    if total_pos == 0 or total_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)], 0.5

    tp = 0
    fp = 0
    points = [(0.0, 0.0)]
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        points.append((fp / total_neg, tp / total_pos))
    points.append((1.0, 1.0))

    points = dedup_points(points)
    auc = trapezoid_auc(points)
    return points, auc


def dedup_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped = []
    for p in points:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return deduped


def trapezoid_auc(points: list[tuple[float, float]]) -> float:
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def build_roc_svg(
    points: list[tuple[float, float]],
    auc: float,
    title: str,
    output_file: Path,
    curve_color: str,
) -> None:
    width, height = 720, 520
    left, right, top, bottom = 85, 40, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + x * plot_w

    def sy(y: float) -> float:
        return top + (1 - y) * plot_h

    grid_lines = []
    for i in range(6):
        v = i / 5
        x = sx(v)
        y = sy(v)
        grid_lines.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}' stroke='#e5e7eb' stroke-width='1'/>")
        grid_lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e5e7eb' stroke-width='1'/>")

    path_d = " ".join(
        [
            ("M" if idx == 0 else "L") + f" {sx(x):.2f} {sy(y):.2f}"
            for idx, (x, y) in enumerate(points)
        ]
    )
    diag = f"M {sx(0):.2f} {sy(0):.2f} L {sx(1):.2f} {sy(1):.2f}"

    labels = []
    for i in range(6):
        v = i / 5
        labels.append(
            f"<text x='{sx(v):.1f}' y='{top + plot_h + 25}' text-anchor='middle' font-size='12' fill='#374151'>{v:.1f}</text>"
        )
        labels.append(
            f"<text x='{left - 15}' y='{sy(v) + 4:.1f}' text-anchor='end' font-size='12' fill='#374151'>{v:.1f}</text>"
        )

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='20' font-weight='700' fill='#111827'>{title}</text>
<text x='{width/2:.1f}' y='50' text-anchor='middle' font-size='13' fill='#4b5563'>AUROC = {auc:.4f}</text>
{''.join(grid_lines)}
<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='none' stroke='#111827' stroke-width='1.5'/>
<path d='{diag}' fill='none' stroke='#AEBED0' stroke-width='2' stroke-dasharray='7 5'/>
<path d='{path_d}' fill='none' stroke='{curve_color}' stroke-width='4' stroke-linejoin='round' stroke-linecap='round'/>
{''.join(labels)}
<text x='{width/2:.1f}' y='{height-20}' text-anchor='middle' font-size='14' fill='#111827'>False Positive Rate</text>
<text x='25' y='{height/2:.1f}' transform='rotate(-90 25 {height/2:.1f})' text-anchor='middle' font-size='14' fill='#111827'>True Positive Rate</text>
</svg>"""
    output_file.write_text(svg, encoding="utf-8")


def build_coef_svg(coef_df: pd.DataFrame, title: str, output_file: Path) -> None:
    pos_df = coef_df[coef_df["direction"] == "positive"].nlargest(6, "coef")[["feature", "coef"]].copy()
    neg_df = coef_df[coef_df["direction"] == "negative"].nsmallest(6, "coef")[["feature", "coef"]].copy()
    plot_df = pd.concat([pos_df, neg_df], ignore_index=True)
    plot_df["short_feature"] = plot_df["feature"].map(shorten_feature)

    min_coef = float(plot_df["coef"].min())
    max_coef = float(plot_df["coef"].max())
    bound = max(abs(min_coef), abs(max_coef), 0.5)

    width = 980
    row_h = 34
    top = 70
    left_label = 320
    chart_left = left_label + 30
    chart_right = width - 55
    chart_w = chart_right - chart_left
    zero_x = chart_left + chart_w / 2
    height = top + len(plot_df) * row_h + 70

    def sx(v: float) -> float:
        return zero_x + (v / (2 * bound)) * chart_w

    rows = []
    for idx, row in plot_df.iterrows():
        y = top + idx * row_h + row_h / 2
        x0 = zero_x
        x1 = sx(float(row["coef"]))
        bar_x = min(x0, x1)
        bar_w = abs(x1 - x0)
        color = pick_state_color(str(row["feature"]))
        rows.append(
            f"<text x='{left_label}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='#111827'>{row['short_feature']}</text>"
            f"<rect x='{bar_x:.1f}' y='{y - 10:.1f}' width='{bar_w:.1f}' height='20' fill='{color}' rx='3'/>"
            f"<text x='{x1 + (8 if row['coef'] >= 0 else -8):.1f}' y='{y + 4:.1f}' text-anchor='{'start' if row['coef'] >= 0 else 'end'}' font-size='11' fill='#374151'>{row['coef']:.3f}</text>"
        )

    ticks = []
    for frac in [-1, -0.5, 0, 0.5, 1]:
        val = frac * bound
        x = sx(val)
        ticks.append(f"<line x1='{x:.1f}' y1='{top-10}' x2='{x:.1f}' y2='{height-45}' stroke='#e5e7eb' stroke-width='1'/>")
        ticks.append(f"<text x='{x:.1f}' y='{height-20}' text-anchor='middle' font-size='12' fill='#374151'>{val:.2f}</text>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{width/2:.1f}' y='30' text-anchor='middle' font-size='20' font-weight='700' fill='#111827'>{title}</text>
<text x='{width/2:.1f}' y='52' text-anchor='middle' font-size='13' fill='#4b5563'>Top positive and negative coefficients</text>
{''.join(ticks)}
<line x1='{zero_x:.1f}' y1='{top-10}' x2='{zero_x:.1f}' y2='{height-45}' stroke='#111827' stroke-width='1.5'/>
{''.join(rows)}
<text x='{chart_left + chart_w/2:.1f}' y='{height-2}' text-anchor='middle' font-size='14' fill='#111827'>Coefficient</text>
</svg>"""
    output_file.write_text(svg, encoding="utf-8")


def pick_state_color(feature: str) -> str:
    feature = str(feature)
    if any(key in feature for key in ["S1", "search", "需求表达"]):
        return STATE_COLORS["S1"]
    if any(key in feature for key in ["S2", "compare", "比较筛选"]):
        return STATE_COLORS["S2"]
    if any(key in feature for key in ["S3", "intent", "意向强化"]):
        return STATE_COLORS["S3"]
    if any(key in feature for key in ["S4", "convert", "转化完成"]):
        return STATE_COLORS["S4"]
    return "#AEBED0"


def shorten_feature(text: str, max_len: int = 34) -> str:
    text = str(text)
    replacements = {
        "cat__": "",
        "num__": "",
        "state_scene_key_": "",
        "state_style_key_": "",
        "scene_key_": "",
        "prev_scene_1_": "prevScene:",
        "prev_style_1_": "prevStyle:",
        "prev_state_1_": "prev1:",
        "prev_state_2_": "prev2:",
        "content_bucket_": "bucket:",
        "meal_period_": "meal:",
        "state_name_": "state:",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def build_metrics_markdown(metrics_df: pd.DataFrame) -> str:
    show = metrics_df[
        [
            "target",
            "accuracy",
            "roc_auc",
            "precision_pos",
            "recall_pos",
            "f1_pos",
            "positive_rate_test",
        ]
    ].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.4f}")
    return to_simple_markdown_table(show)


def to_simple_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def write_summary_md(metrics_df: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Logistic 回归报告资产",
            "",
            "## 模型指标",
            build_metrics_markdown(metrics_df),
            "",
            "## 生成图表",
            "- `roc_curve_deepen.svg`：未来3步进入更深状态模型的 AUROC 曲线",
            "- `roc_curve_convert.svg`：未来3步转化模型的 AUROC 曲线",
            "- `coef_deepen.svg`：进入更深状态模型的关键正负系数",
            "- `coef_convert.svg`：短期转化模型的关键正负系数",
        ]
    )
    (ASSET_DIR / "report_asset_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_excel(INPUT_FILE, sheet_name="02_model_metrics")
    deepen_coef_df = pd.read_excel(INPUT_FILE, sheet_name="05_deepen_coef")
    convert_coef_df = pd.read_excel(INPUT_FILE, sheet_name="09_convert_coef")
    deepen_pred_df = pd.read_excel(INPUT_FILE, sheet_name="06_deepen_predictions")
    convert_pred_df = pd.read_excel(INPUT_FILE, sheet_name="10_convert_predictions")

    deepen_points, deepen_auc = roc_points(deepen_pred_df["true_y"].tolist(), deepen_pred_df["pred_prob"].tolist())
    convert_points, convert_auc = roc_points(convert_pred_df["true_y"].tolist(), convert_pred_df["pred_prob"].tolist())

    build_roc_svg(
        deepen_points,
        deepen_auc,
        "未来3步进入更深状态 - ROC Curve",
        ASSET_DIR / "roc_curve_deepen.svg",
        curve_color=STATE_COLORS["S3"],
    )
    build_roc_svg(
        convert_points,
        convert_auc,
        "未来3步转化 - ROC Curve",
        ASSET_DIR / "roc_curve_convert.svg",
        curve_color=STATE_COLORS["S4"],
    )
    build_coef_svg(deepen_coef_df, "进入更深状态模型 - 关键系数", ASSET_DIR / "coef_deepen.svg")
    build_coef_svg(convert_coef_df, "未来3步转化模型 - 关键系数", ASSET_DIR / "coef_convert.svg")
    write_summary_md(metrics_df)

    print(f"Saved assets to: {ASSET_DIR}")


if __name__ == "__main__":
    main()
