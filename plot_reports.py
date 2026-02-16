"""
Read evaluation reports for a given emotion representation id and create
a plotnine plot with two facets: Accuracy and Macro F1 vs context window,
with one line per context method.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from configuration import EMOTION_REPRESENTATIONS
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_wrap,
    labs,
    theme_minimal,
    theme,
    element_text,
    element_rect,
    element_blank,
    scale_x_continuous,
    scale_color_manual,
)


REPORTS_DIR = Path(__file__).resolve().parent / "reports"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def parse_report(path: Path) -> dict | None:
    """Extract Accuracy, Macro F1, Context Window, and Context Method from a report file."""
    text = path.read_text()
    out = {}

    acc_match = re.search(r"Accuracy\s+:\s+([\d.]+)%", text)
    f1_match = re.search(r"Macro F1\s+:\s+([\d.]+)%", text)
    cw_match = re.search(r"Context Window\s+:\s+(\d+)", text)
    cm_match = re.search(r"Context Method\s+:\s+(.+)", text)

    if not (acc_match and f1_match and cw_match):
        return None

    out["accuracy"] = float(acc_match.group(1))
    out["macro_f1"] = float(f1_match.group(1))
    out["context_window"] = int(cw_match.group(1))
    out["context_method"] = cm_match.group(1).strip() if cm_match else "no context"

    return out


def load_reports(er_id: int) -> pd.DataFrame:
    """Load all reports for emotion representation id from directories named er{er_id}_*."""
    rows = []

    # Single file for context window 0: er{er_id}_cw0_report.txt
    cw0_path = REPORTS_DIR / f"er{er_id}_cw0_report.txt"
    if cw0_path.exists():
        parsed = parse_report(cw0_path)
        if parsed:
            rows.append(parsed)

    # Directories whose name starts with er{er_id}_ (e.g. er0_cm0, er0_cm1)
    for subdir in REPORTS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith(f"er{er_id}_"):
            continue
        for report_path in subdir.glob("*.txt"):
            if "_report.txt" not in report_path.name:
                continue
            parsed = parse_report(report_path)
            if parsed:
                rows.append(parsed)

    if not rows:
        raise FileNotFoundError(
            f"No reports found for emotion representation id {er_id} in {REPORTS_DIR}. "
            f"Expected directories named er{er_id}_* (e.g. er{er_id}_cm0, er{er_id}_cm1) "
            f"and/or file er{er_id}_cw0_report.txt."
        )

    return pd.DataFrame(rows)


COLORS = ["#2b8cbe", "#e34a33", "#31a354"]


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the dataframe and connect the cw=0 baseline to each context method's line."""
    context_methods = sorted(
        m for m in df["context_method"].unique() if m != "no context"
    )

    baseline = df[df["context_method"] == "no context"]
    if not baseline.empty:
        extras = []
        for cm in context_methods:
            row = baseline.iloc[0].copy()
            row["context_method"] = cm
            extras.append(row)
        df = pd.concat([df[df["context_method"] != "no context"], pd.DataFrame(extras)], ignore_index=True)

    long = df.melt(
        id_vars=["context_window", "context_method"],
        value_vars=["accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].replace(
        {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    )
    long["context_method"] = long["context_method"].str.title()
    return long


def plot_metrics(df: pd.DataFrame, er_id: int, output_path: Path | None = None):
    """Create a two-facet plot: Accuracy and Macro F1 vs context window, lines = context method."""
    er_description = EMOTION_REPRESENTATIONS.get(
        er_id, f"Emotion representation {er_id}"
    )
    title_text = f"{er_description}: Accuracy and Macro F1 by context window".title()

    long = _prepare_data(df)
    context_methods = sorted(long["context_method"].unique())
    color_map = dict(zip(context_methods, COLORS[: len(context_methods)]))

    cw_values = sorted(long["context_window"].unique())

    p = (
        ggplot(long, aes(x="context_window", y="value", color="context_method"))
        + geom_line(size=1.2)
        + geom_point(size=3)
        + facet_wrap("metric", scales="free_y", ncol=2)
        + scale_x_continuous(breaks=cw_values)
        + scale_color_manual(values=color_map)
        + labs(
            x="Context Window",
            y="Score (%)",
            color="Context Method",
            title=title_text,
        )
        + theme_minimal()
        + theme(
            figure_size=(10, 5),
            plot_title=element_text(size=13, ha="center"),
            axis_title_x=element_text(size=11, margin={"t": 10}),
            axis_title_y=element_text(size=11, margin={"r": 10}),
            axis_text=element_text(size=10),
            strip_text=element_text(size=11, weight="bold"),
            strip_background=element_rect(fill="#e8e8e8", color="#cccccc"),
            legend_position="bottom",
            legend_title=element_text(size=10, weight="bold"),
            legend_text=element_text(size=9),
            legend_background=element_rect(fill="white", color="#dddddd"),
            panel_spacing_x=0.04,
            plot_background=element_rect(fill="white", color="none"),
            panel_grid_minor=element_blank(),
        )
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_with_strip_colors(p, output_path, dpi=150)
        print(f"Saved plot to {output_path}")

    return p


def _apply_strip_colors(plot):
    """Color facet strips: Accuracy=green, Macro F1=blue (using plotnine theme targets)."""
    strip_colors = ["#a8ddb5", "#b4d4ed"]  # soft green, soft blue
    strips = getattr(plot.theme.targets, "strip_background_x", [])
    for i, rect in enumerate(strips):
        if i < len(strip_colors):
            rect.set_facecolor(strip_colors[i])
            rect.set_edgecolor("#cccccc")


def _save_with_strip_colors(plot, path: Path, dpi: int = 150):
    """Draw the plot, color facet strips, then save."""
    fig = plot.draw(show=False)
    _apply_strip_colors(plot)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy and macro F1 from reports for an emotion representation id."
    )
    parser.add_argument(
        "er_id",
        type=int,
        help="Emotion representation id (reports are read from directories er{er_id}_*).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot. Default: plots/er<id>_metrics_plot.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot (requires display).",
    )
    args = parser.parse_args()

    df = load_reports(args.er_id)
    print(f"Loaded {len(df)} reports for emotion representation {args.er_id}.")
    print(df.to_string(index=False))

    output = args.output or PLOTS_DIR / f"er{args.er_id}_metrics_plot.png"
    p = plot_metrics(df, args.er_id, output_path=output)

    if args.show:
        fig = p.draw(show=False)
        _apply_strip_colors(p)
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
