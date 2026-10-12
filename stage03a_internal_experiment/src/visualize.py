from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data_pipeline import sha256_file
from pipeline import write_json


WIDTH, HEIGHT = 1600, 1000
INK = (34, 48, 68)
MUTED = (98, 112, 129)
GRID = (222, 228, 235)
BLUE = (54, 113, 190)
RED = (215, 87, 75)
GREEN = (40, 146, 111)
PURPLE = (130, 94, 180)
ORANGE = (231, 151, 55)
PALE = (244, 247, 250)


def _font_path() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No supported Chinese font found; install Noto Sans CJK or configure a Windows CJK font")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=0)


def _centered(draw: ImageDraw.ImageDraw, center_x: float, y: int, text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def _new_canvas(title: str, font_file: Path) -> tuple[Image.Image, ImageDraw.ImageDraw, dict[str, ImageFont.FreeTypeFont]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    fonts = {"title": _font(font_file, 36), "subtitle": _font(font_file, 22), "axis": _font(font_file, 20),
             "tick": _font(font_file, 16), "small": _font(font_file, 15), "number": _font(font_file, 29)}
    _centered(draw, WIDTH / 2, 28, title, fonts["title"], INK)
    _centered(draw, WIDTH / 2, 75, "数据范围：AFQMC 内部 tune（阈值亦在此集选择；不是最终独立验证）", fonts["subtitle"], MUTED)
    return image, draw, fonts


def _axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x_label: str, y_label: str,
          font: dict[str, ImageFont.FreeTypeFont], y_max: float = 1.0, y_percent: bool = False) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="white", outline=INK, width=2)
    for step in range(6):
        value = y_max * step / 5
        y = int(y1 - (y1 - y0) * step / 5)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{value:.0%}" if y_percent else f"{value:.2f}" if y_max <= 1.0 else f"{value:.0f}"
        draw.text((x0 - 74, y - 12), label, font=font["tick"], fill=MUTED)
    _centered(draw, (x0 + x1) / 2, y1 + 46, x_label, font["axis"], INK)
    # Draw the vertical label with a rotated transparent layer.
    label_layer = Image.new("RGBA", (260, 38), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((2, 2), y_label, font=font["axis"], fill=INK)
    label_layer = label_layer.rotate(90, expand=True)
    draw._image.paste(label_layer, (max(0, x0 - 120), int((y0 + y1 - label_layer.height) / 2)), label_layer)
    return box


def _read_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["official_split"] != "train" or raw["internal_split"] != "tune":
                raise ValueError("plot source contains a row outside official Train/internal tune")
            row: dict[str, Any] = dict(raw)
            row["label"] = int(raw["label"])
            row["prediction"] = int(raw["prediction"])
            row["score"] = float(raw["score"]) if raw["score"] != "" else None
            row["threshold"] = float(raw["threshold"]) if raw["threshold"] != "" else None
            grouped[row["method"]].append(row)
    return grouped


def _read_scan(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        result = []
        for row in csv.DictReader(handle):
            result.append({key: (float(row[key]) if key == "threshold" else float(row[key]) if key in {"precision", "recall", "f1", "accuracy"} else row[key] == "True" if key == "is_selected" else int(row[key])) for key in row})
    return result


def _distribution_figure(grouped: dict[str, list[dict[str, Any]]], output: Path, font_file: Path) -> None:
    image, draw, fonts = _new_canvas("内部 tune 相似度分布", font_file)
    panels = [(130, 210, 770, 790), (875, 210, 1515, 790)]
    bins = 40
    for (method, title), box in zip((("count_char2", "字符 2-gram 词频 + 余弦"), ("tfidf_char2", "字符 2-gram TF-IDF + 余弦")), panels):
        x0, y0, x1, y1 = box
        rows = grouped[method]
        class_counts = {label: Counter() for label in (0, 1)}
        totals = Counter(row["label"] for row in rows)
        for row in rows:
            index = min(bins - 1, max(0, int(row["score"] * bins)))
            class_counts[row["label"]][index] += 1
        max_rate = max(class_counts[label][i] / max(1, totals[label]) for label in (0, 1) for i in range(bins))
        scale = max(0.12, max_rate * 1.12)
        _axes(draw, box, "余弦分数", "类内比例（每类归一）", fonts, scale, y_percent=True)
        _centered(draw, (x0 + x1) / 2, y0 - 57, title, fonts["axis"], INK)
        draw.rectangle((x0 + 25, y0 - 28, x0 + 47, y0 - 13), fill=BLUE)
        draw.text((x0 + 55, y0 - 34), "标签 0（不相似）", font=fonts["small"], fill=INK)
        draw.rectangle((x0 + 255, y0 - 28, x0 + 277, y0 - 13), fill=RED)
        draw.text((x0 + 285, y0 - 34), "标签 1（相似）", font=fonts["small"], fill=INK)
        for i in range(bins):
            center = x0 + (i + 0.5) * (x1 - x0) / bins
            group_w = (x1 - x0) / bins * 0.76
            for label, offset, color in ((0, -group_w / 4, BLUE), (1, group_w / 4, RED)):
                rate = class_counts[label][i] / max(1, totals[label])
                h = rate / scale * (y1 - y0)
                xx = center + offset
                draw.rectangle((int(xx - group_w / 4), int(y1 - h), int(xx + group_w / 4), y1), fill=color)
        for tick in range(6):
            value = tick / 5
            xx = int(x0 + value * (x1 - x0))
            draw.line((xx, y1, xx, y1 + 6), fill=INK, width=1)
            _centered(draw, xx, y1 + 9, f"{value:.1f}", fonts["tick"], MUTED)
    image.save(output, format="PNG", optimize=True)


def _threshold_figure(scans: dict[str, list[dict[str, Any]]], selected: dict[str, float], output: Path, font_file: Path) -> None:
    image, draw, fonts = _new_canvas("内部 tune 阈值与指标关系", font_file)
    panels = [(130, 180, 770, 800), (875, 180, 1515, 800)]
    metric_colors = {"precision": BLUE, "recall": GREEN, "f1": RED}
    for (method, title), box in zip((("count_char2", "词频 + 余弦"), ("tfidf_char2", "TF-IDF + 余弦")), panels):
        x0, y0, x1, y1 = box
        _axes(draw, box, "阈值", "指标", fonts, 1.0)
        _centered(draw, (x0 + x1) / 2, y0 - 43, title, fonts["axis"], INK)
        rows = scans[method]
        for tick in range(6):
            value = tick / 5
            xx = int(x0 + value * (x1 - x0))
            draw.line((xx, y1, xx, y1 + 6), fill=INK, width=1)
            _centered(draw, xx, y1 + 9, f"{value:.1f}", fonts["tick"], MUTED)
        for metric, color in metric_colors.items():
            points = []
            for row in rows:
                tx = min(1.0, max(0.0, row["threshold"]))
                xx = int(x0 + tx * (x1 - x0))
                yy = int(y1 - row[metric] * (y1 - y0))
                points.append((xx, yy))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
        t = min(1.0, max(0.0, selected[method]))
        tx = int(x0 + t * (x1 - x0))
        for yy in range(y0, y1, 14):
            draw.line((tx, yy, tx, min(yy + 7, y1)), fill=PURPLE, width=2)
        legend_x = x0 + 35
        for index, (metric, color) in enumerate(metric_colors.items()):
            lx = legend_x + index * 175
            draw.line((lx, y0 - 15, lx + 30, y0 - 15), fill=color, width=4)
            draw.text((lx + 38, y0 - 27), {"precision": "Precision", "recall": "Recall", "f1": "F1"}[metric], font=fonts["small"], fill=INK)
        draw.line((x1 - 120, y0 - 15, x1 - 90, y0 - 15), fill=PURPLE, width=2)
        draw.text((x1 - 84, y0 - 27), "选定阈值", font=fonts["small"], fill=INK)
    image.save(output, format="PNG", optimize=True)


def _comparison_figure(grouped: dict[str, list[dict[str, Any]]], output: Path, font_file: Path) -> None:
    image, draw, fonts = _new_canvas("内部 tune 方案指标对照", font_file)
    box = (170, 200, 1450, 765)
    x0, y0, x1, y1 = box
    _axes(draw, box, "", "指标", fonts, 1.0)
    for tick in range(6):
        value = tick / 5
        xx = int(x0 + value * (x1 - x0))
        draw.line((xx, y1, xx, y1 + 6), fill=INK, width=1)
        _centered(draw, xx, y1 + 9, f"{value:.1f}", fonts["tick"], MUTED)
    methods = [("count_char2", "词频+余弦", BLUE), ("tfidf_char2", "TF-IDF+余弦", RED), ("majority_fit_baseline", "多数类基线", GREEN)]
    metrics = ("precision", "recall", "f1", "accuracy")
    metric_colors = (BLUE, RED, GREEN, ORANGE)
    group_width = (x1 - x0) / len(methods)
    bar_width = min(45, group_width / 7)
    for mi, (method, label, _) in enumerate(methods):
        rows = grouped[method]
        y = [int(row["label"]) for row in rows]
        pred = [int(row["prediction"]) for row in rows]
        tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
        fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
        fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
        tn = sum(a == 0 and b == 0 for a, b in zip(y, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(y)
        vals = (precision, recall, f1, accuracy)
        center = x0 + (mi + 0.5) * group_width
        start = center - 2 * bar_width
        for j, (metric, color, value) in enumerate(zip(metrics, metric_colors, vals)):
            xx = start + j * bar_width * 1.12
            yy = y1 - value * (y1 - y0)
            draw.rounded_rectangle((int(xx), int(yy), int(xx + bar_width), y1), radius=5, fill=color)
            _centered(draw, xx + bar_width / 2, max(y0 - 28, int(yy) - 28), f"{value:.3f}", fonts["small"], INK)
        _centered(draw, center, y1 + 35, label, fonts["axis"], INK)
    for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
        lx = 400 + j * 210
        draw.rectangle((lx, 125, lx + 22, 145), fill=color)
        draw.text((lx + 30, 120), {"precision": "Precision", "recall": "Recall", "f1": "F1", "accuracy": "Accuracy"}[metric], font=fonts["small"], fill=INK)
    image.save(output, format="PNG", optimize=True)


def _confusion_figure(rows: list[dict[str, Any]], output: Path, font_file: Path) -> None:
    image, draw, fonts = _new_canvas("TF-IDF 内部 tune 混淆矩阵", font_file)
    _centered(draw, WIDTH / 2, 145, "正类：相似问句（标签 1）", fonts["axis"], INK)
    x0, y0 = 470, 260
    cell_w, cell_h = 330, 250
    counts = Counter((int(row["label"]), int(row["prediction"])) for row in rows)
    max_count = max(counts.values()) or 1
    colors = [[(232, 242, 252), (252, 232, 229)], [(238, 248, 242), (226, 243, 235)]]
    for true in (0, 1):
        for pred in (0, 1):
            x = x0 + pred * cell_w
            y = y0 + true * cell_h
            value = counts[(true, pred)]
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=colors[true][pred], outline=(255, 255, 255), width=4)
            _centered(draw, x + cell_w / 2, y + 55, str(value), fonts["number"], INK)
            title = {(0, 0): "TN｜正确判为不相似", (0, 1): "FP｜误报为相似", (1, 0): "FN｜漏检相似问句", (1, 1): "TP｜正确判为相似"}[(true, pred)]
            _centered(draw, x + cell_w / 2, y + 130, title, fonts["axis"], INK)
    _centered(draw, x0 + cell_w, y0 - 43, "预测标签", fonts["axis"], INK)
    _centered(draw, x0 + cell_w / 2, y0 - 9, "0：不相似", fonts["small"], MUTED)
    _centered(draw, x0 + cell_w * 1.5, y0 - 9, "1：相似", fonts["small"], MUTED)
    _centered(draw, x0 - 100, y0 + cell_h, "真实标签", fonts["axis"], INK)
    draw.text((x0 - 160, y0 + 95), "0：不相似", font=fonts["small"], fill=MUTED)
    draw.text((x0 - 160, y0 + cell_h + 95), "1：相似", font=fonts["small"], fill=MUTED)
    total = sum(counts.values())
    _centered(draw, WIDTH / 2, 825, f"总样本数 N = {total}；来源：内部 tune 行级预测 CSV", fonts["subtitle"], MUTED)
    image.save(output, format="PNG", optimize=True)


def generate_all_figures(experiment_dir: Path) -> dict[str, Any]:
    output_dir = experiment_dir / "outputs"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    font_file = _font_path()
    # Check that the selected font actually provides glyphs for Chinese labels.
    test_font = _font(font_file, 24)
    if test_font.getmask("中文相似问句").getbbox() is None:
        raise ValueError(f"selected font has no Chinese glyphs: {font_file}")

    prediction_path = output_dir / "internal_tune_predictions.csv"
    grouped = _read_predictions(prediction_path)
    required = {"count_char2", "tfidf_char2", "majority_fit_baseline"}
    if set(grouped) != required:
        raise ValueError(f"unexpected tune prediction methods: {sorted(grouped)}")
    scans = {
        "count_char2": _read_scan(output_dir / "threshold_scan_count_char2.csv"),
        "tfidf_char2": _read_scan(output_dir / "threshold_scan_tfidf_char2.csv"),
    }
    selected = {
        method: next(float(row["threshold"]) for row in scan if row["is_selected"])
        for method, scan in scans.items()
    }
    outputs = {
        "score_distribution": figure_dir / "internal_tune_score_distribution.png",
        "threshold_metrics": figure_dir / "internal_tune_threshold_metrics.png",
        "method_comparison": figure_dir / "internal_tune_method_comparison.png",
        "tfidf_confusion_matrix": figure_dir / "internal_tune_tfidf_confusion_matrix.png",
    }
    _distribution_figure(grouped, outputs["score_distribution"], font_file)
    _threshold_figure(scans, selected, outputs["threshold_metrics"], font_file)
    _comparison_figure(grouped, outputs["method_comparison"], font_file)
    _confusion_figure(grouped["tfidf_char2"], outputs["tfidf_confusion_matrix"], font_file)

    return {
        "data_scope": "official Train/internal tune only; not final independent validation",
        "source_prediction_csv": prediction_path.relative_to(experiment_dir).as_posix(),
        "source_prediction_csv_sha256": sha256_file(prediction_path),
        "source_threshold_csv_sha256": {
            method: sha256_file(output_dir / f"threshold_scan_{method}.csv") for method in scans
        },
        "font_path": str(font_file),
        "font_sha256": sha256_file(font_file),
        "font_chinese_glyph_check": "中文相似问句 rendered using selected font",
        "figures": {
            name: {"path": path.relative_to(experiment_dir).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in outputs.items()
        },
        "selected_thresholds": selected,
    }
