"""Create the four publication figures for the mixed-precision square-root article."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

PAPER_SETS = ("I", "II", "III", "IV", "V", "VI", "VII")
CONDITIONS = (4.0, 100.0)
ORDER = 128
EPSILON64 = np.finfo(np.float64).eps

FORMAT_SPECS: dict[str, dict[str, Any]] = {
    "fp32": {
        "label": "FP32",
        "bits": "32 bits",
        "color": "#202124",
        "marker": "o",
    },
    "bf16": {
        "label": "BF16",
        "bits": "16 bits",
        "color": "#2a9d8f",
        "marker": "^",
    },
    "fp8_e4m3fn": {
        "label": "FP8 E4M3FN",
        "bits": "8 bits",
        "color": "#1677b8",
        "marker": "s",
    },
    "mxfp4_e2m1": {
        "label": "MXFP4 E2M1",
        "bits": "4.25 bits",
        "color": "#e56b1f",
        "marker": "D",
    },
    "fp4_e2m1": {
        "label": "FP4 E2M1",
        "bits": "4 bits",
        "color": "#a72d3d",
        "marker": "X",
    },
}

KERNEL_FORMAT_SPECS: dict[str, dict[str, str]] = {
    "fp32": {
        "label": "FP32",
        "color": "#202124",
        "marker": "o",
    },
    "bf16": {
        "label": "BF16",
        "color": "#0072b2",
        "marker": "^",
    },
    "fp8_e4m3fn": {
        "label": "FP8 E4M3FN",
        "color": "#009e73",
        "marker": "s",
    },
    "mxfp8_e4m3fn": {
        "label": "MXFP8 E4M3FN",
        "color": "#56b4e9",
        "marker": "v",
    },
    "mxfp4_e2m1": {
        "label": "MXFP4 E2M1",
        "color": "#d55e00",
        "marker": "D",
    },
    "fp4_e2m1": {
        "label": "FP4 E2M1",
        "color": "#cc79a7",
        "marker": "P",
    },
}

FRONTIER_FORMATS = (
    ("fp32", "FP32"),
    ("bf16", "BF16"),
    ("mxfp8_e5m2", "MXFP8 E5M2"),
    ("mxfp8_e4m3fn", "MXFP8 E4M3FN"),
    ("fp8_e5m2", "FP8 E5M2"),
    ("fp8_e4m3fn", "FP8 E4M3FN"),
    ("fp8_e4m3fn_log_s", "FP8 E4M3FN + log spectrum"),
    ("mxfp4_e2m1", "MXFP4 E2M1"),
    ("fp4_e2m1", "FP4 E2M1"),
    ("fp4_e2m1_log_s", "FP4 E2M1 + log spectrum"),
)

LOW_PRECISION_FORMATS = (
    {
        "directory": "bf16",
        "label": "BF16 basis / BF16 spectrum",
        "short_label": "BF16",
        "spectrum": "bf16",
        "color": "#555555",
        "marker": "o",
    },
    {
        "directory": "fp8_e4m3fn_log_s",
        "label": "FP8 basis / log-FP8 spectrum",
        "short_label": "FP8 E4M3 + log spectrum",
        "spectrum": "fp8_e4m3fn_log_s",
        "color": "#0072b2",
        "marker": "s",
    },
    {
        "directory": "mxfp8_e4m3fn",
        "label": "BF16 basis / MXFP8 E4M3 spectrum",
        "short_label": "MXFP8 E4M3",
        "spectrum": "mxfp8_e4m3fn",
        "color": "#009e73",
        "marker": "^",
    },
    {
        "directory": "mxfp8_e5m2",
        "label": "BF16 basis / MXFP8 E5M2 spectrum",
        "short_label": "MXFP8 E5M2",
        "spectrum": "mxfp8_e5m2",
        "color": "#56b4e9",
        "marker": "v",
    },
    {
        "directory": "mxfp4_e2m1",
        "label": "BF16 basis / MXFP4 E2M1 spectrum",
        "short_label": "MXFP4 E2M1",
        "spectrum": "mxfp4_e2m1",
        "color": "#d55e00",
        "marker": "D",
    },
)

TIMING_FORMATS = tuple(
    spec
    for spec in LOW_PRECISION_FORMATS
    if spec["directory"] in {"bf16", "mxfp8_e4m3fn", "mxfp8_e5m2", "mxfp4_e2m1"}
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication plots from recorded matrix-square-root benchmarks."
    )
    parser.add_argument(
        "--paper-dir", type=Path, default=Path("metal_paper_benchmark_1024")
    )
    parser.add_argument(
        "--precision-dir", type=Path, default=Path("metal_benchmark")
    )
    parser.add_argument(
        "--kernel-dir", type=Path, default=Path("metal_kernel_benchmark")
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("artifacts/article_diagnostics"),
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=Path("artifacts/article_timing_v2"),
    )
    parser.add_argument(
        "--dispatch-dir",
        type=Path,
        default=Path("artifacts/article_dispatch_v2"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/article_showcase")
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.nan


def integer(row: dict[str, str], field: str) -> int:
    value = number(row, field)
    return int(value) if math.isfinite(value) else 0


def is_success(row: dict[str, str]) -> bool:
    return (
        row.get("converged") == "1"
        and row.get("finite") == "1"
        and math.isfinite(number(row, "final_forward_error"))
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "normal",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "font.size": 9,
            "font.weight": "normal",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def load_paper_data(
    paper_dir: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        test_set: {
            "summary": read_rows(paper_dir / test_set / "summary.csv"),
            "iterations": read_rows(paper_dir / test_set / "iterations.csv"),
        }
        for test_set in PAPER_SETS
    }


def load_precision_data(
    precision_dir: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        format_name: {
            "summary": read_rows(precision_dir / format_name / "summary.csv"),
            "iterations": read_rows(precision_dir / format_name / "iterations.csv"),
        }
        for format_name in FORMAT_SPECS
    }


def paper_accuracy_plot(
    paper_data: dict[str, dict[str, list[dict[str, str]]]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), sharey=True)
    flat_axes = axes.ravel()
    residual_color = "#146c94"
    error_color = "#d65f2e"

    for index, test_set in enumerate(PAPER_SETS):
        axis = flat_axes[index]
        rows = sorted(
            paper_data[test_set]["iterations"], key=lambda row: integer(row, "iteration")
        )
        steps = [integer(row, "iteration") for row in rows]
        residuals = [number(row, "residual") for row in rows]
        errors = [number(row, "forward_error") for row in rows]
        order = integer(rows[0], "order")

        axis.semilogy(
            steps,
            residuals,
            color=residual_color,
            marker="o",
            markersize=4,
            linewidth=2.0,
        )
        axis.semilogy(
            steps,
            errors,
            color=error_color,
            linestyle="--",
            marker="s",
            markersize=3.6,
            linewidth=1.8,
        )
        axis.axhline(EPSILON64, color="#777777", linestyle=":", linewidth=1.0)
        axis.set_title(f"Set {test_set}  ·  n={order}")
        axis.set_xlabel("Correction step")
        axis.set_ylim(8e-17, 5e-4)
        axis.grid(True, which="major")
        if index % 4 == 0:
            axis.set_ylabel("Relative error")

    flat_axes[-1].axis("off")
    legend = [
        Line2D([0], [0], color=residual_color, marker="o", linewidth=2, label="Residual"),
        Line2D(
            [0],
            [0],
            color=error_color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Forward error",
        ),
        Line2D([0], [0], color="#777777", linestyle=":", label="FP64 unit roundoff"),
    ]
    flat_axes[-1].legend(handles=legend, loc="center", fontsize=11)
    figure.suptitle(
        "Gao–Kressner–Shao accuracy tests, reproduced with Apple Accelerate",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        0.965,
        "FP32 Schur factors and correction system · FP64 residual and update",
        ha="center",
        color="#555555",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(figure, output_dir / "01_paper_accuracy_replication.png")


def paper_runtime_plot(
    paper_data: dict[str, dict[str, list[dict[str, str]]]], output_dir: Path
) -> Path:
    phase_fields = (
        ("single_seconds", "FP32 initial solve", "#277da1"),
        ("residual_seconds", "FP64 residual", "#f9c74f"),
        ("correction_system_seconds", "FP32 correction system", "#43aa8b"),
        ("updates_and_others_seconds", "FP64 update and other", "#f3722c"),
    )
    normalized: dict[str, list[float]] = {field: [] for field, _, _ in phase_fields}
    totals: list[float] = []
    labels: list[str] = []

    for test_set in PAPER_SETS:
        row = paper_data[test_set]["iterations"][0]
        reference = number(row, "reference_timing_mean_seconds")
        values = [number(row, field) / reference for field, _, _ in phase_fields]
        for (field, _, _), value in zip(phase_fields, values, strict=True):
            normalized[field].append(value)
        totals.append(sum(values))
        labels.append(f"{test_set}\nn={integer(row, 'order')}")

    figure, axis = plt.subplots(figsize=(11.5, 6.1))
    positions = np.arange(len(PAPER_SETS))
    bottom = np.zeros(len(PAPER_SETS))
    for field, label, color in phase_fields:
        values = np.asarray(normalized[field])
        axis.bar(
            positions,
            values,
            bottom=bottom,
            width=0.72,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values

    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.3)
    axis.text(6.47, 1.015, "fixed FP64", ha="right", va="bottom", color="#333333")
    for position, total in zip(positions, totals, strict=True):
        label = f"{(1.0 / total):.2f}× faster" if total < 1 else f"{total:.2f}× slower"
        axis.text(position, total + 0.07, label, ha="center", va="bottom", fontsize=9)

    axis.set_xticks(positions, labels)
    axis.set_ylabel("Time relative to fixed FP64 Accelerate")
    axis.set_xlabel("Paper test set")
    axis.set_ylim(0, max(totals) * 1.16)
    axis.grid(True, axis="y")
    axis.legend(loc="upper left", ncols=2)
    axis.set_title(
        "The FP32 factorization saves time only when refinement stays cheap",
        fontsize=16,
        pad=38,
    )
    figure.text(
        0.5,
        0.925,
        "Same phase decomposition and seven matrix families as the paper",
        ha="center",
        color="#555555",
        fontsize=11,
    )
    figure.tight_layout()
    return save_figure(figure, output_dir / "02_paper_runtime_replication.png")


def matching_rows(
    rows: list[dict[str, str]], order: int, condition: float
) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if integer(row, "order") == order
            and math.isclose(number(row, "condition"), condition)
        ),
        key=lambda row: integer(row, "iteration"),
    )


def matching_summary(
    rows: list[dict[str, str]], order: int, condition: float
) -> dict[str, str]:
    matches = matching_rows(rows, order, condition)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary row for order={order}, condition={condition}; "
            f"found {len(matches)}."
        )
    return matches[0]


def low_bit_recovery_plot(
    precision_data: dict[str, dict[str, list[dict[str, str]]]], output_dir: Path
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex="col", sharey="row")
    plotted_formats = ("fp32", "bf16", "fp8_e4m3fn", "mxfp4_e2m1", "fp4_e2m1")

    for column, condition in enumerate(CONDITIONS):
        for format_name in plotted_formats:
            spec = FORMAT_SPECS[format_name]
            rows = matching_rows(
                precision_data[format_name]["iterations"], ORDER, condition
            )
            summary = matching_summary(
                precision_data[format_name]["summary"], ORDER, condition
            )
            steps = [integer(row, "iteration") for row in rows]
            residuals = [number(row, "residual") for row in rows]
            errors = [number(row, "forward_error") for row in rows]

            if is_success(summary):
                axes[0, column].semilogy(
                    steps,
                    residuals,
                    color=spec["color"],
                    marker=spec["marker"],
                    markevery=max(1, len(steps) // 8),
                    markersize=4,
                    linewidth=2.0,
                )
                axes[1, column].semilogy(
                    steps,
                    errors,
                    color=spec["color"],
                    marker=spec["marker"],
                    markevery=max(1, len(steps) // 8),
                    markersize=4,
                    linewidth=2.0,
                )
            else:
                axes[0, column].scatter(
                    [0],
                    [number(summary, "initial_residual")],
                    color=spec["color"],
                    marker="X",
                    s=75,
                    zorder=5,
                )
                axes[1, column].scatter(
                    [0],
                    [number(summary, "initial_forward_error")],
                    color=spec["color"],
                    marker="X",
                    s=75,
                    zorder=5,
                )
                axes[1, column].annotate(
                    "rank deficient",
                    (0, number(summary, "initial_forward_error")),
                    xytext=(8, -15),
                    textcoords="offset points",
                    color=spec["color"],
                    fontsize=9,
                )

        axes[0, column].set_title(rf"Target condition $\kappa={condition:g}$")
        axes[1, column].set_xlabel("Correction step")
        for row in range(2):
            axes[row, column].grid(True, which="major")
            axes[row, column].axhline(
                EPSILON64, color="#777777", linestyle=":", linewidth=1.0
            )

    axes[0, 0].set_ylabel("Relative residual")
    axes[1, 0].set_ylabel("Relative forward error")
    axes[0, 0].set_ylim(8e-17, 2.0)
    axes[1, 0].set_ylim(8e-17, 2.0)
    axes[0, 0].set_xlim(-0.7, 33)
    axes[0, 1].set_xlim(-0.7, 33)

    legend = [
        Line2D(
            [0],
            [0],
            color=FORMAT_SPECS[name]["color"],
            marker=FORMAT_SPECS[name]["marker"],
            linewidth=2,
            label=f"{FORMAT_SPECS[name]['label']} · {FORMAT_SPECS[name]['bits']}",
        )
        for name in plotted_formats
    ]
    figure.legend(handles=legend, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(
        "Lower factor precision increases the refinement cost",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.text(
        0.5,
        0.958,
        "Order 128 · FP32/BF16 correction or reconstructed FP32 correction · FP64 residual and update",
        ha="center",
        color="#555555",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    return save_figure(figure, output_dir / "03_low_bit_recovery_ladder.png")


def recovery_arrows_plot(
    precision_data: dict[str, dict[str, list[dict[str, str]]]], output_dir: Path
) -> Path:
    format_names = ("fp32", "bf16", "fp8_e4m3fn", "mxfp4_e2m1", "fp4_e2m1")
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.5), sharey=True)

    for axis, condition in zip(axes, CONDITIONS, strict=True):
        for position, format_name in enumerate(format_names):
            spec = FORMAT_SPECS[format_name]
            row = matching_summary(
                precision_data[format_name]["summary"], ORDER, condition
            )
            initial = number(row, "initial_forward_error")
            final = number(row, "final_forward_error")

            axis.scatter(
                [position],
                [initial],
                color=spec["color"],
                marker=spec["marker"],
                s=75,
                zorder=4,
            )
            if is_success(row):
                axis.annotate(
                    "",
                    xy=(position, final),
                    xytext=(position, initial),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": spec["color"],
                        "linewidth": 2.2,
                        "mutation_scale": 13,
                    },
                )
                axis.scatter(
                    [position],
                    [final],
                    facecolor="white",
                    edgecolor=spec["color"],
                    marker=spec["marker"],
                    linewidth=1.8,
                    s=72,
                    zorder=5,
                )
                midpoint = math.sqrt(initial * final)
                axis.annotate(
                    f"{integer(row, 'iterations')} corrections",
                    (position, midpoint),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    color=spec["color"],
                    fontsize=9,
                )
            else:
                axis.scatter(
                    [position],
                    [initial],
                    facecolor="none",
                    edgecolor="#a72d3d",
                    marker="o",
                    linewidth=1.8,
                    s=130,
                    zorder=3,
                )
                axis.annotate(
                    "no correction\n(rank deficient)",
                    (position, initial),
                    xytext=(0, -31),
                    textcoords="offset points",
                    ha="center",
                    color="#a72d3d",
                    fontsize=9,
                )

        axis.axhline(EPSILON64, color="#777777", linestyle=":", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_ylim(8e-17, 2.0)
        axis.set_xticks(
            range(len(format_names)),
            [
                f"{FORMAT_SPECS[name]['label']}\n{FORMAT_SPECS[name]['bits']}"
                for name in format_names
            ],
        )
        axis.set_title(rf"Target condition $\kappa={condition:g}$")
        axis.grid(True, axis="y", which="major")
        axis.set_xlabel("Frozen-factor storage")

    axes[0].set_ylabel("Relative forward error")
    figure.suptitle(
        "What refinement can repair—and what it cannot",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    figure.text(
        0.5,
        0.95,
        "Filled marker: stored initial root · open marker: refined result · order 128",
        ha="center",
        color="#555555",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(figure, output_dir / "04_refinement_repair_limit.png")


def precision_frontier_hero(precision_dir: Path, output_dir: Path) -> Path:
    orders = (16, 32, 64, 96, 128)
    conditions = (4.0, 100.0, 10_000.0, 1_000_000.0)
    font_size = 9
    lower_limit = 9.0
    upper_limit = 15.0
    summaries = {
        format_name: read_rows(precision_dir / format_name / "summary.csv")
        for format_name, _ in FRONTIER_FORMATS
    }
    rows_by_format = {
        format_name: {
            (integer(row, "order"), number(row, "condition")): row
            for row in rows
        }
        for format_name, rows in summaries.items()
    }
    figure, axes = plt.subplots(1, len(orders), figsize=(15.5, 6.0), sharey=True)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#eeeeee")
    image = None

    for panel_index, (axis, order) in enumerate(zip(axes, orders, strict=True)):
        values = np.full((len(FRONTIER_FORMATS), len(conditions)), np.nan)
        for row_index, (format_name, _) in enumerate(FRONTIER_FORMATS):
            for column_index, condition in enumerate(conditions):
                row = rows_by_format[format_name].get((order, condition))
                if row is None or not is_success(row):
                    continue
                error = number(row, "final_forward_error")
                if error > 0.0:
                    values[row_index, column_index] = -math.log10(error)

        image = axis.imshow(
            values,
            cmap=colormap,
            vmin=lower_limit,
            vmax=upper_limit,
            aspect="auto",
            interpolation="none",
        )
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if math.isfinite(value):
                    text = f"{value:.1f}"
                    color = "white" if value < 11.5 else "#222222"
                else:
                    text = "×"
                    color = "#555555"
                axis.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=font_size,
                    fontweight="normal",
                )

        axis.set_title(
            f"Order {order}",
            fontsize=font_size,
            fontweight="normal",
            pad=6,
        )
        axis.set_xticks(
            range(len(conditions)),
            ("4", r"$10^2$", r"$10^4$", r"$10^6$"),
        )
        axis.set_yticks(
            range(len(FRONTIER_FORMATS)),
            [label for _, label in FRONTIER_FORMATS],
        )
        axis.tick_params(
            axis="both",
            labelsize=font_size,
            labelleft=panel_index == 0,
            length=3,
        )
        axis.set_xticks(
            np.arange(-0.5, len(conditions), 1.0),
            minor=True,
        )
        axis.set_yticks(
            np.arange(-0.5, len(FRONTIER_FORMATS), 1.0),
            minor=True,
        )
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)

    figure.subplots_adjust(
        left=0.17,
        right=0.91,
        bottom=0.14,
        top=0.88,
        wspace=0.18,
    )
    colorbar_axis = figure.add_axes((0.93, 0.14, 0.012, 0.74))
    if image is None:
        raise RuntimeError("The precision-frontier data is empty.")
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        ticks=np.arange(lower_limit, upper_limit + 1.0),
    )
    colorbar.set_label(
        r"$-\log_{10}(\mathrm{final\ relative\ forward\ error})$",
        fontsize=font_size,
    )
    colorbar.ax.tick_params(labelsize=font_size)
    figure.suptitle(
        "Final forward error by factor-storage format and target condition number",
        fontsize=font_size,
        fontweight="normal",
        y=0.965,
    )
    figure.supxlabel("Target condition number", fontsize=font_size, y=0.055)
    figure.text(
        0.5,
        0.015,
        "×  No finite converged result",
        ha="center",
        va="bottom",
        fontsize=font_size,
        fontweight="normal",
        color="#555555",
    )
    return save_figure(
        figure, output_dir / "00_hero_precision_frontier.png"
    )


def metal_speedup_hero(kernel_dir: Path, output_dir: Path) -> Path:
    condition = 4.0
    font_size = 10
    confidence_multiplier = 1.96
    failed_marker_y = 0.62
    rows_by_path = {
        kernel_path: {
            format_name: read_rows(
                kernel_dir / kernel_path / format_name / "summary.csv"
            )
            for format_name in KERNEL_FORMAT_SPECS
        }
        for kernel_path in ("legacy", "tiled")
    }
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    maximum_upper = 1.0
    has_failed_cases = False

    for format_name, spec in KERNEL_FORMAT_SPECS.items():
        legacy = {
            (integer(row, "order"), number(row, "condition")): row
            for row in rows_by_path["legacy"][format_name]
        }
        tiled = {
            (integer(row, "order"), number(row, "condition")): row
            for row in rows_by_path["tiled"][format_name]
        }
        common_orders = sorted(
            {
                key[0]
                for key in set(legacy).intersection(tiled)
                if math.isclose(key[1], condition)
            }
        )
        orders: list[int] = []
        ratios: list[float] = []
        confidence_intervals: list[float] = []
        failed_orders: list[int] = []
        for order in common_orders:
            legacy_row = legacy[(order, condition)]
            tiled_row = tiled[(order, condition)]
            if not is_success(legacy_row) or not is_success(tiled_row):
                failed_orders.append(order)
                continue
            legacy_time = number(legacy_row, "mixed_total_seconds")
            tiled_time = number(tiled_row, "mixed_total_seconds")
            ratio = legacy_time / tiled_time
            legacy_variance = number(legacy_row, "timing_variance_seconds2")
            tiled_variance = number(tiled_row, "timing_variance_seconds2")
            legacy_evaluations = integer(legacy_row, "evaluations")
            tiled_evaluations = integer(tiled_row, "evaluations")
            ratio_variance = ratio * ratio * (
                legacy_variance
                / (legacy_evaluations * legacy_time * legacy_time)
                + tiled_variance
                / (tiled_evaluations * tiled_time * tiled_time)
            )
            half_width = confidence_multiplier * math.sqrt(
                max(0.0, ratio_variance)
            )
            orders.append(order)
            ratios.append(ratio)
            confidence_intervals.append(half_width)
            maximum_upper = max(maximum_upper, ratio + half_width)

        axis.errorbar(
            orders,
            ratios,
            yerr=confidence_intervals,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=1.3,
            markersize=5,
            capsize=2,
            label=spec["label"],
        )
        if failed_orders:
            has_failed_cases = True
            axis.scatter(
                failed_orders,
                [failed_marker_y] * len(failed_orders),
                color=spec["color"],
                marker="x",
                s=32,
                linewidth=1.2,
            )

    axis.axhline(
        1.0,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Equal mean time",
    )
    if has_failed_cases:
        failed_handle = Line2D(
            [0],
            [0],
            color="#666666",
            marker="x",
            linestyle="none",
            markersize=5,
            label="Did not converge",
        )
        handles, labels = axis.get_legend_handles_labels()
        handles.append(failed_handle)
        labels.append("Did not converge")
        axis.legend(
            handles,
            labels,
            loc="upper right",
            ncols=2,
            fontsize=font_size,
        )
    else:
        axis.legend(loc="upper right", ncols=2, fontsize=font_size)

    axis.set_xlim(12, 132)
    axis.set_ylim(0.5, maximum_upper * 1.08)
    axis.set_xticks((16, 32, 64, 96, 128))
    axis.set_xlabel("Matrix order", fontsize=font_size)
    axis.set_ylabel(
        "Mean end-to-end time ratio (legacy / tiled)", fontsize=font_size
    )
    axis.tick_params(axis="both", labelsize=font_size)
    axis.grid(True, which="major")
    axis.set_title(
        r"Execution-time ratio for converged cases at $\kappa=4$ "
        r"(95% CI, 100 evaluations per path)",
        fontsize=font_size,
        fontweight="normal",
        pad=8,
    )
    figure.tight_layout()
    return save_figure(figure, output_dir / "00_hero_metal_speedup.png")


def load_diagnostic_summaries(
    root: Path,
) -> dict[str, list[dict[str, str]]]:
    return {
        str(spec["directory"]): read_rows(
            root / str(spec["directory"]) / "summary.csv"
        )
        for spec in LOW_PRECISION_FORMATS
    }


def low_precision_frontier_plot(
    diagnostic_dir: Path, output_dir: Path
) -> Path:
    order = 128
    conditions = (4.0, 100.0, 10_000.0, 1_000_000.0)
    summaries = load_diagnostic_summaries(diagnostic_dir)
    values = np.full((len(LOW_PRECISION_FORMATS), len(conditions)), np.nan)
    states = np.full(values.shape, "failed", dtype=object)

    for row_index, spec in enumerate(LOW_PRECISION_FORMATS):
        rows = summaries[str(spec["directory"])]
        for column_index, condition in enumerate(conditions):
            row = next(
                candidate
                for candidate in rows
                if integer(candidate, "order") == order
                and math.isclose(number(candidate, "condition"), condition)
            )
            selected = row.get("selected_spectrum_stage", "")
            if selected.endswith("_rejected_to_fp32"):
                states[row_index, column_index] = "fallback"
            elif is_success(row):
                states[row_index, column_index] = "retained"
                error = number(row, "final_forward_error")
                if error > 0.0:
                    values[row_index, column_index] = -math.log10(error)

    figure, axis = plt.subplots(figsize=(8.6, 3.5))
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#f2f2f2")
    image = axis.imshow(
        values,
        cmap=colormap,
        vmin=9.0,
        vmax=15.0,
        interpolation="none",
        aspect="auto",
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            state = states[row_index, column_index]
            value = values[row_index, column_index]
            if state == "fallback":
                axis.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1.0,
                        1.0,
                        facecolor="#d9d9d9",
                        edgecolor="none",
                    )
                )
                text = "FP32"
                color = "#333333"
            elif state == "failed":
                text = "×"
                color = "#555555"
            else:
                text = f"{value:.1f}"
                color = "white" if value < 11.5 else "#222222"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=color,
                fontweight="normal",
            )

    axis.set_xticks(
        range(len(conditions)),
        ("4", r"$10^2$", r"$10^4$", r"$10^6$"),
    )
    axis.set_yticks(
        range(len(LOW_PRECISION_FORMATS)),
        [str(spec["label"]) for spec in LOW_PRECISION_FORMATS],
    )
    axis.set_xlabel("Target condition number")
    axis.set_title(
        r"Order 128; cell value is $-\log_{10}$ final forward error",
        fontweight="normal",
    )
    axis.set_xticks(np.arange(-0.5, len(conditions), 1.0), minor=True)
    axis.set_yticks(
        np.arange(-0.5, len(LOW_PRECISION_FORMATS), 1.0), minor=True
    )
    axis.grid(which="minor", color="white", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"$-\log_{10}$ relative forward error")
    axis.legend(
        handles=[
            Patch(facecolor="#d9d9d9", label="Spectrum replaced by FP32"),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#555555",
                linestyle="none",
                label="No finite converged result",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
    )
    figure.subplots_adjust(left=0.28, right=0.91, top=0.88, bottom=0.30)
    return save_figure(
        figure, output_dir / "00_low_precision_frontier.png"
    )


def spectral_diagnostics_plot(
    diagnostic_dir: Path, output_dir: Path
) -> Path:
    conditions = (4.0, 100.0, 10_000.0, 1_000_000.0)
    orders = (16, 32, 64, 96, 128)
    summaries = load_diagnostic_summaries(diagnostic_dir)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))

    basis_specs = (
        (
            LOW_PRECISION_FORMATS[0],
            "BF16 basis (used by BF16, MXFP8, and MXFP4)",
        ),
        (LOW_PRECISION_FORMATS[1], "FP8 E4M3 basis"),
    )
    for spec, label in basis_specs:
        rows = summaries[str(spec["directory"])]
        orthogonality = []
        for order in orders:
            row = next(
                candidate
                for candidate in rows
                if integer(candidate, "order") == order
                and math.isclose(number(candidate, "condition"), 4.0)
            )
            orthogonality.append(number(row, "quantized_basis_orthogonality"))
        axes[0].plot(
            orders,
            orthogonality,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=1.4,
            markersize=5,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(orders)
    axes[0].set_xlabel("Matrix order")
    axes[0].set_ylabel(r"$\|Q^\mathsf{T}Q-I\|_F$")
    axes[0].set_title("Stored-basis orthogonality", fontweight="normal")
    axes[0].grid(True, which="major")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=1
    )

    for spec in LOW_PRECISION_FORMATS:
        rows = summaries[str(spec["directory"])]
        errors = []
        fallback_conditions = []
        fallback_errors = []
        for condition in conditions:
            row = next(
                candidate
                for candidate in rows
                if integer(candidate, "order") == 128
                and math.isclose(number(candidate, "condition"), condition)
            )
            error = number(row, "initial_forward_error")
            fallback = row.get("selected_spectrum_stage", "").endswith(
                "_rejected_to_fp32"
            )
            errors.append(math.nan if fallback else error)
            if fallback:
                fallback_conditions.append(condition)
                fallback_errors.append(error)
        axes[1].plot(
            conditions,
            errors,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=1.4,
            markersize=5,
            label=spec["short_label"],
        )
        if fallback_conditions:
            axes[1].scatter(
                fallback_conditions,
                fallback_errors,
                facecolors="white",
                edgecolors=spec["color"],
                marker=spec["marker"],
                s=45,
                linewidth=1.2,
                zorder=4,
            )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        conditions,
        ("4", r"$10^2$", r"$10^4$", r"$10^6$"),
    )
    axes[1].set_xlabel("Target condition number")
    axes[1].set_ylabel("Initial relative forward error")
    axes[1].set_title("Frozen-factor error, order 128", fontweight="normal")
    axes[1].grid(True, which="major")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#555555",
            linestyle="none",
        )
    )
    labels.append("Open marker: FP32 spectrum fallback")
    axes[1].legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncols=2,
    )

    figure.subplots_adjust(
        left=0.08, right=0.98, top=0.91, bottom=0.30, wspace=0.24
    )
    return save_figure(
        figure, output_dir / "01_basis_spectrum_diagnostics.png"
    )


def metal_timing_breakdown_plot(
    timing_dir: Path, dispatch_dir: Path, output_dir: Path
) -> Path:
    order = 128
    condition = 4.0
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    component_specs = (
        ("factorization_seconds", "FP32 eigensolve", "#4c78a8"),
        ("factor_quantization_seconds", "Factor quantization", "#72b7b2"),
        ("initial_root_remainder", "Other root setup", "#b5bd68"),
        ("residual_seconds", "FP64 residual", "#f2cf5b"),
        ("correction_system_seconds", "FP32 correction", "#e17c05"),
        ("updates_and_others_seconds", "FP64 update and other", "#b279a2"),
    )
    positions = np.arange(len(TIMING_FORMATS))
    bottom = np.zeros(len(TIMING_FORMATS))
    for field, label, color in component_specs:
        values = []
        for spec in TIMING_FORMATS:
            rows = read_rows(
                timing_dir / str(spec["directory"]) / "summary.csv"
            )
            row = next(
                candidate
                for candidate in rows
                if integer(candidate, "order") == order
                and math.isclose(number(candidate, "condition"), condition)
            )
            if field == "initial_root_remainder":
                seconds = max(
                    0.0,
                    number(row, "initial_root_seconds")
                    - number(row, "factor_quantization_seconds"),
                )
            else:
                seconds = number(row, field)
            values.append(1_000.0 * seconds)
        axes[0].bar(
            positions,
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            label=label,
        )
        bottom += np.asarray(values)
    axes[0].set_xticks(
        positions,
        [str(spec["short_label"]) for spec in TIMING_FORMATS],
        rotation=20,
        ha="right",
    )
    axes[0].set_ylabel("Mean end-to-end time, ms")
    axes[0].set_title(
        r"Phase time, order 128, $\kappa=4$, 100 evaluations",
        fontweight="normal",
    )
    axes[0].grid(True, axis="y")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=2
    )

    dispatch_styles = (
        ("per_stage", "One command buffer per dispatch", "#d55e00", "s"),
        ("fused", "One command buffer per correction", "#009e73", "o"),
    )
    dispatch_count = 0
    for policy, label, color, marker in dispatch_styles:
        rows = read_rows(
            dispatch_dir / policy / "mxfp8_e4m3fn" / "summary.csv"
        )
        orders = []
        latencies = []
        for row in sorted(rows, key=lambda candidate: integer(candidate, "order")):
            if not math.isclose(number(row, "condition"), condition):
                continue
            iterations = integer(row, "iterations")
            orders.append(integer(row, "order"))
            latencies.append(
                1_000.0
                * number(row, "correction_system_seconds")
                / iterations
            )
            dispatch_count = integer(row, "kernel_dispatches_per_correction")
        axes[1].plot(
            orders,
            latencies,
            color=color,
            marker=marker,
            linewidth=1.4,
            markersize=5,
            label=label,
        )
    axes[1].set_xticks((32, 64, 128))
    axes[1].set_xlabel("Matrix order")
    axes[1].set_ylabel("Mean time per correction, ms")
    axes[1].set_title(
        f"Native MXFP8 E4M3 correction; {dispatch_count} dispatches",
        fontweight="normal",
    )
    axes[1].grid(True)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=1
    )

    figure.subplots_adjust(
        left=0.08, right=0.98, top=0.91, bottom=0.31, wspace=0.24
    )
    return save_figure(
        figure, output_dir / "02_metal_timing_breakdown.png"
    )


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    paths = [
        low_precision_frontier_plot(
            arguments.diagnostic_dir, arguments.output_dir
        ),
        spectral_diagnostics_plot(
            arguments.diagnostic_dir, arguments.output_dir
        ),
        metal_timing_breakdown_plot(
            arguments.timing_dir,
            arguments.dispatch_dir,
            arguments.output_dir,
        ),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
