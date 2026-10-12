"""Create four aggregate-only official Dev plots from the frozen output files."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STAGE3B = Path(__file__).resolve().parent
OUTPUT = STAGE3B / "outputs"
FIGURES = OUTPUT / "figures"
STAGE3A_METRICS = ROOT / "stage03a_internal_experiment" / "outputs" / "metrics_summary.json"
WIDTH, HEIGHT = 1600, 1000
INK = (34, 48, 68)
MUTED = (94, 108, 126)
GRID = (222, 228, 235)
BLUE = (54, 113, 190)
RED = (215, 87, 75)
GREEN = (40, 146, 111)
PALE = (244, 247, 250)


def font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_metrics() -> dict[str, Any]:
    return json.loads((OUTPUT / "dev_metrics.json").read_text(encoding="utf-8"))


def read_predictions() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (OUTPUT / "dev_predictions_local_only.csv").open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            grouped[raw["method"]].append({
                "label": int(raw["label"]), "score": float(raw["score"]), "prediction": int(raw["prediction"]),
            })
    return grouped


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 36), title, font=font(38), fill=INK)
    draw.text((70, 94), subtitle, font=font(20), fill=MUTED)
    return image, draw


def footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.rectangle((50, HEIGHT - 88, WIDTH - 50, HEIGHT - 40), fill=PALE)
    draw.text((68, HEIGHT - 76), text, font=font(17), fill=MUTED)


def save(image: Image.Image, filename: str) -> dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    image.save(path, format="PNG", optimize=True)
    return {
        "path": path.relative_to(STAGE3B).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": image.width,
        "height": image.height,
    }


def metric_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    image, draw = canvas("Official Dev: frozen-threshold metric comparison", "Positive label: 1 | Primary method: TF-IDF char-2 + cosine | Fixed Count char-2 baseline")
    values = metrics["metrics"]
    keys = ["f1", "accuracy", "false_positive_rate", "specificity", "balanced_accuracy"]
    labels = ["F1", "Accuracy", "FPR", "Specificity", "Balanced accuracy"]
    left, top, right, bottom = 180, 205, 1450, 810
    draw.rectangle((left, top, right, bottom), outline=INK, width=2)
    for step in range(6):
        y = bottom - (bottom - top) * step // 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((100, y - 12), f"{step / 5:.1f}", font=font(18), fill=MUTED)
    group_width = (right - left) / len(keys)
    for index, (key, label) in enumerate(zip(keys, labels, strict=True)):
        center = left + group_width * (index + 0.5)
        for offset, method, color in ((-34, "count_char2", BLUE), (34, "tfidf_char2", RED)):
            value = float(values[method][key])
            bar_h = int((bottom - top) * value)
            x0, x1 = int(center + offset - 20), int(center + offset + 20)
            draw.rectangle((x0, bottom - bar_h, x1, bottom), fill=color)
            draw.text((x0 - 10, bottom - bar_h - 30), f"{value:.3f}", font=font(15), fill=INK)
        draw.text((int(center - 60), bottom + 18), label, font=font(17), fill=INK)
    draw.rectangle((470, 160, 495, 182), fill=BLUE)
    draw.text((505, 158), "Count", font=font(18), fill=INK)
    draw.rectangle((620, 160, 645, 182), fill=RED)
    draw.text((655, 158), "TF-IDF", font=font(18), fill=INK)
    n = values["tfidf_char2"]["sample_count"]
    footer(draw, f"official Dev | threshold source: frozen internal tune | score >= threshold | N={n} per method | FPR shown as a cost metric")
    return save(image, "dev_metrics_comparison.png")


def tfidf_confusion(metrics: dict[str, Any]) -> dict[str, Any]:
    row = metrics["metrics"]["tfidf_char2"]
    image, draw = canvas("Official Dev: TF-IDF confusion matrix", "TF-IDF character 2-gram + paired cosine | positive label: 1")
    tn, fp, fn, tp = row["tn"], row["fp"], row["fn"], row["tp"]
    cells = ((tn, "Predicted 0", "Actual 0"), (fp, "Predicted 1", "Actual 0"),
             (fn, "Predicted 0", "Actual 1"), (tp, "Predicted 1", "Actual 1"))
    box = (340, 230, 1260, 770)
    mid_x, mid_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    vals = ((tn, fp), (fn, tp))
    max_val = max(tn, fp, fn, tp, 1)
    for r in range(2):
        for c in range(2):
            val = vals[r][c]
            shade = int(244 - 110 * val / max_val)
            x0, y0 = box[0] + c * (box[2] - box[0]) // 2, box[1] + r * (box[3] - box[1]) // 2
            x1, y1 = x0 + (box[2] - box[0]) // 2, y0 + (box[3] - box[1]) // 2
            draw.rectangle((x0, y0, x1, y1), fill=(shade, min(255, shade + 8), 255), outline="white", width=6)
            draw.text((x0 + 130, y0 + 95), f"{val:,}", font=font(54), fill=INK)
    draw.text((box[0] + 200, box[1] - 45), "Predicted 0", font=font(22), fill=INK)
    draw.text((box[0] + 650, box[1] - 45), "Predicted 1", font=font(22), fill=INK)
    draw.text((box[0] - 190, box[1] + 190), "Actual 0", font=font(22), fill=INK)
    draw.text((box[0] - 190, box[1] + 460), "Actual 1", font=font(22), fill=INK)
    footer(draw, f"official Dev | threshold {row['threshold']:.17g} from frozen internal tune | score >= threshold | positive=1 | N={row['sample_count']}")
    return save(image, "dev_tfidf_confusion_matrix.png")


def score_distributions(predictions: dict[str, list[dict[str, Any]]], metrics: dict[str, Any]) -> dict[str, Any]:
    image, draw = canvas("Official Dev: cosine score distributions by true label", "Each panel uses all rows for one method and one true label; bars show within-group proportions")
    panels = [("count_char2", 0, BLUE), ("count_char2", 1, RED),
              ("tfidf_char2", 0, BLUE), ("tfidf_char2", 1, RED)]
    bins = 24
    positions = [(90, 195), (835, 195), (90, 515), (835, 515)]
    for (method, label, color), (x, y) in zip(panels, positions, strict=True):
        rows = [row for row in predictions[method] if row["label"] == label]
        counts = [0] * bins
        for row in rows:
            index = min(bins - 1, int(float(row["score"]) * bins))
            counts[index] += 1
        max_prop = max(counts) / max(len(rows), 1)
        panel_w, panel_h = 660, 245
        draw.text((x, y - 35), f"{method} | true label={label} | n={len(rows)}", font=font(18), fill=INK)
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=INK, width=2)
        for tick in range(5):
            xx = x + tick * panel_w // 4
            draw.line((xx, y, xx, y + panel_h), fill=GRID, width=1)
            draw.text((xx - 10, y + panel_h + 5), f"{tick / 4:.2f}", font=font(14), fill=MUTED)
        step = panel_w / bins
        for index, count in enumerate(counts):
            bar_h = int(panel_h * (count / max(len(rows), 1)) / max(max_prop, 1e-12))
            x0 = int(x + index * step + 1)
            x1 = int(x + (index + 1) * step - 1)
            draw.rectangle((x0, y + panel_h - bar_h, x1, y + panel_h), fill=color)
    row = metrics["metrics"]["tfidf_char2"]
    footer(draw, f"official Dev | thresholds from frozen internal tune | score >= threshold | positive=1 | N={row['sample_count']} per method; no Dev tuning")
    return save(image, "dev_score_distributions.png")


def tune_vs_dev(metrics: dict[str, Any]) -> dict[str, Any]:
    tune = json.loads(STAGE3A_METRICS.read_text(encoding="utf-8"))
    image, draw = canvas("F1 comparison: internal tune and official Dev", "The same frozen method-specific threshold is used on Dev; tune thresholds were selected on tune")
    left, top, right, bottom = 220, 210, 1400, 790
    draw.rectangle((left, top, right, bottom), outline=INK, width=2)
    for step in range(6):
        y = bottom - (bottom - top) * step // 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((140, y - 12), f"{step / 5:.1f}", font=font(18), fill=MUTED)
    methods = ["count_char2", "tfidf_char2"]
    group_w = (right - left) / 2
    for i, method in enumerate(methods):
        center = left + group_w * (i + 0.5)
        values = [float(tune["methods"][method]["f1"]), float(metrics["metrics"][method]["f1"])]
        for offset, value, color in ((-38, values[0], BLUE), (38, values[1], GREEN)):
            h = int((bottom - top) * value)
            x0, x1 = int(center + offset - 24), int(center + offset + 24)
            draw.rectangle((x0, bottom - h, x1, bottom), fill=color)
            draw.text((x0 - 25, bottom - h - 30), f"{value:.3f}", font=font(17), fill=INK)
        draw.text((int(center - 100), bottom + 18), method, font=font(20), fill=INK)
    draw.rectangle((500, 160, 525, 182), fill=BLUE)
    draw.text((535, 158), "Internal tune", font=font(18), fill=INK)
    draw.rectangle((700, 160, 725, 182), fill=GREEN)
    draw.text((735, 158), "Official Dev", font=font(18), fill=INK)
    n_tune = tune["methods"]["tfidf_char2"]["sample_count"]
    n_dev = metrics["metrics"]["tfidf_char2"]["sample_count"]
    footer(draw, f"tune: thresholds selected here | Dev: thresholds frozen from tune | positive=1 | n tune={n_tune}, n Dev={n_dev} per method")
    return save(image, "internal_tune_vs_dev_f1.png")


def generate_all() -> dict[str, Any]:
    metrics = read_metrics()
    predictions = read_predictions()
    figures = {
        "dev_metric_comparison": metric_comparison(metrics),
        "dev_tfidf_confusion_matrix": tfidf_confusion(metrics),
        "dev_score_distributions": score_distributions(predictions, metrics),
        "internal_tune_vs_dev_f1": tune_vs_dev(metrics),
    }
    record = {
        "source_metrics_sha256": hashlib.sha256((OUTPUT / "dev_metrics.json").read_bytes()).hexdigest(),
        "source_predictions_sha256": hashlib.sha256((OUTPUT / "dev_predictions_local_only.csv").read_bytes()).hexdigest(),
        "figures": figures,
        "all_figures_aggregate_only": True,
        "row_level_question_text_in_figures": False,
    }
    (OUTPUT / "plot_rebuild_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(generate_all(), ensure_ascii=False, indent=2))
