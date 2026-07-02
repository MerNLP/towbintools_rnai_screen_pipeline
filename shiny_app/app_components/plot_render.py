"""Plotly scatter helpers for the RNAi screen dashboard."""

import numpy as np
import plotly.graph_objects as go
import polars as pl

# Later traces win on click; controls listed last.
CLICK_PRIORITY_ORDER = ["EV", "empty", "ama-1", "mex-3", "hit", "non-hit"]
PLOT_MARKER_SIZE = 7
MARKER_OPACITY_BY_COLOR = {
    "non-hit": 0.45,
    "hit": 0.82,
}


def empty_fig(msg: str = "Load a CSV to get started"):
    fig = go.FigureWidget()
    fig.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    return fig


def pick_plot_row_at_click(
    plot_df: pl.DataFrame,
    x: float,
    y: float,
    xcol: str,
    ycol: str,
    *,
    color_col: str = "plot_color",
    max_dist_frac: float = 0.035,
) -> dict | None:
    """Nearest click target; prefer controls over hits/non-hits."""
    if plot_df is None or plot_df.height == 0:
        return None
    work = plot_df.filter(
        pl.col(xcol).is_finite() & pl.col(ycol).is_finite()
    )
    if work.height == 0:
        return None

    x_f, y_f = float(x), float(y)
    work = work.with_columns(
        ((pl.col(xcol) - pl.lit(x_f)) ** 2 + (pl.col(ycol) - pl.lit(y_f)) ** 2).alias(
            "_d2"
        )
    )
    bounds = work.select(
        pl.col(xcol).min().alias("xmin"),
        pl.col(xcol).max().alias("xmax"),
        pl.col(ycol).min().alias("ymin"),
        pl.col(ycol).max().alias("ymax"),
    ).row(0, named=True)
    span = max(
        float(bounds["xmax"]) - float(bounds["xmin"]),
        float(bounds["ymax"]) - float(bounds["ymin"]),
        1e-9,
    )
    # Minimum radius so controls stay clickable in dense regions.
    min_click_radius = 0.12
    max_dist = max(max_dist_frac * span, min_click_radius)
    max_d2 = max_dist**2

    near = work.filter(pl.col("_d2") <= max_d2)
    if near.height == 0:
        return work.sort("_d2").head(1).drop("_d2").row(0, named=True)

    if color_col in near.columns:
        for cat in CLICK_PRIORITY_ORDER:
            sub = near.filter(pl.col(color_col) == cat)
            if sub.height > 0:
                return sub.sort("_d2").head(1).drop("_d2").row(0, named=True)

    return near.sort("_d2").head(1).drop("_d2").row(0, named=True)


def _hover_display_cols(
    sub,
    x_col: str,
    y_col: str,
    hover_cols: list[str],
) -> list[str]:
    """Column order for hover: x/y first, then extra metadata."""
    cols: list[str] = []
    for c in (x_col, y_col):
        if c in sub.columns and c not in cols:
            cols.append(c)
    for c in hover_cols:
        if c in sub.columns and c not in cols:
            cols.append(c)
    return cols


def _hover_text(
    sub,
    cols: list[str],
    *,
    hover_labels: dict[str, str] | None = None,
) -> list[str]:
    """Build per-point hover strings (sub is a pandas DataFrame slice)."""
    hover_labels = hover_labels or {}
    n = len(sub)
    if not cols:
        return [""] * n
    lines: list[str] = []
    for row in sub[cols].itertuples(index=False, name=None):
        if len(cols) == 1:
            c = cols[0]
            lines.append(f"{hover_labels.get(c, c)}: {row[0]}")
        else:
            lines.append(
                "<br>".join(
                    f"{hover_labels.get(c, c)}: {v}" for c, v in zip(cols, row)
                )
            )
    return lines


def build_layered_scatter(
    pdf,
    x_col: str,
    y_col: str,
    *,
    color_col: str,
    color_map: dict[str, str],
    color_order: list[str],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    hover_cols: list[str] | None = None,
    hover_labels: dict[str, str] | None = None,
    template: str = "simple_white",
) -> go.FigureWidget:
    """Scatter with explicit trace layering so controls stay above the dense cloud."""
    fig = go.FigureWidget()
    hover_cols = hover_cols or []
    hover_labels = hover_labels or {}

    for cat in color_order:
        sub = pdf[pdf[color_col] == cat] if color_col in pdf.columns else pdf.iloc[0:0]
        if sub.empty:
            continue
        display_cols = _hover_display_cols(sub, x_col, y_col, hover_cols)
        marker = dict(
            size=PLOT_MARKER_SIZE,
            color=color_map.get(cat, "#999"),
            opacity=MARKER_OPACITY_BY_COLOR.get(cat, 0.8),
        )
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=str(cat),
                marker=marker,
                text=_hover_text(sub, display_cols, hover_labels=hover_labels),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        template=template,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=56, b=110),
    )
    return fig


def sanitize_xy_pdf(pdf, x_col: str, y_col: str):
    # shinywidgets JSON cannot serialize NaN/Inf.
    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    pdf = pdf.dropna(subset=[x_col, y_col])
    return pdf


def _sanitize_obj(v):
    # Recursively convert NaN/Inf to None for JSON-safe widget payloads.
    if isinstance(v, (list, tuple)):
        return [_sanitize_obj(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_sanitize_obj(x) for x in v.tolist()]
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    return v


def sanitize_figure_for_json(fig):
    for tr in fig.data:
        for attr in ("x", "y", "customdata", "text"):
            if hasattr(tr, attr):
                try:
                    cur = getattr(tr, attr)
                    if cur is not None:
                        setattr(tr, attr, _sanitize_obj(cur))
                except Exception:
                    pass
    return fig
