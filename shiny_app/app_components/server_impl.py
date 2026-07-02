"""Shiny server: load combined CSVs, plot hits, preview images."""

import base64
import hashlib
import io
import os
import re
import threading
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats
from shiny import reactive, render, ui
from shinywidgets import render_widget
from app_components.helpers import (
    _resolve_existing_path,
    apply_library_filter,
    cap_image_rows,
    canonical_control_gene_expr,
    collapse_to_gene_level,
    default_metric_col,
    display_file_col,
    df_has_meaningful_day,
    file_col,
    find_combined_csvs,
    gene_level_key_cols,
    gene_name_match_expr,
    guess_col_among,
    is_plot_synthetic_control_row,
    infer_experiment_dir_from_df,
    library_group_expr,
    library_group_from_code,
    list_dir,
    metric_col_choices,
    meaningful_screen_days,
    numeric_cols,
    preload_join_key_cols,
    prepare_comparison_metrics_df,
    screen_group_keys,
    select_normalization_reference_df,
    selected_metric_col,
    well384_label,
)
from app_components.images import (
    cache_key as _cache_key,
    image_backend_name,
    mask_path_to_raw,
    overlay_mask_on_raw_to_base64_jpeg,
    tiff_to_base64_jpeg,
    PREVIEW_MAX_WIDTH,
)
from app_components.plot_render import (
    build_layered_scatter,
    empty_fig,
    pick_plot_row_at_click,
    sanitize_figure_for_json,
    sanitize_xy_pdf,
)
from app_components.preload import (
    N_PRELOAD_THREADS,
    cache_lock as _cache_lock,
    cache_stats as _cache_stats,
    image_cache as _image_cache,
    preload_threads as _preload_threads,
    preload_variants_for_row as _preload_variants_for_row,
    preload_worker as _preload_worker,
)

UM_PER_PX_DEFAULT = 1.625   # 4× objective, 6.5 µm camera pixel
EPS_LOG = 1.0
DEFAULT_BROWSE_ROOT = Path("/mnt/towbin.data/shared")

MAX_AUTO_PRELOAD_IMAGES = 200
MAX_PRELOAD_PER_PLOTTED_POINT = 8

PLOT_COLOR_MAP = {
    "hit": "#e63946",
    "non-hit": "#f4a261",
    "EV": "#457b9d",
    "empty": "#2a9d8f",
    "mex-3": "#7209b7",
    "ama-1": "#0096c7",
}

# Plot trace order: non-hits in back, controls on top.
PLOT_COLOR_ORDER = ["non-hit", "hit", "EV", "empty", "ama-1", "mex-3"]


def plot_color_expr() -> pl.Expr:
    """Legend / marker color: hit, non-hit, or control subtype from gene_name."""
    is_ctrl = pl.col("is_control").cast(pl.Utf8).str.to_lowercase() == "true"
    return (
        pl.when(is_ctrl)
        .then(canonical_control_gene_expr())
        .when(pl.col("label") == "hit")
        .then(pl.lit("hit"))
        .otherwise(pl.lit("non-hit"))
    )


def server(input, output, session):

    raw_df = reactive.value(None)
    clicked_row = reactive.value(None)
    flagged_mask_paths: reactive.Value[set[str]] = reactive.value(set())
    flag_input_map: reactive.Value[dict[str, str]] = reactive.value({})
    _last_flag_selection: reactive.Value[set[str]] = reactive.value(set())
    _plot_df_cache = reactive.value(None)
    _plot_axes_cache = reactive.value((None, None))
    _plot_fig_cache = reactive.value(None)
    _displayed_images: reactive.Value[list[tuple[str, str, str]]] = reactive.value([])
    browser_cwd = reactive.value(Path.home())
    browser_target = reactive.value("dir")
    preload_started = reactive.value(0)

    def _make_browser_modal():
        cwd = browser_cwd()
        dirs, csvs = list_dir(cwd)
        choices = {}
        for d in dirs:
            choices[str(d)] = f"📁  {d.name}"
        for f in csvs:
            choices[str(f)] = f"📄  {f.name}"

        return ui.modal(
            ui.p(f"📍 {cwd}", style="font-size:0.85em; color:#555; word-break:break-all;"),
            ui.input_select("browser_items", "Contents",
                            choices=choices if choices else {"": "(empty)"},
                            size=12, width="100%"),
            title="Browse for CSV file",
            size="l",
            footer=ui.div(
                ui.input_action_button("browser_up", "↑ Up",
                                       class_="btn btn-sm btn-outline-secondary me-2"),
                ui.input_action_button("browser_open", "Open / Navigate",
                                       class_="btn btn-sm btn-outline-primary me-2"),
                ui.input_action_button("browser_select", "Select this file",
                                       class_="btn btn-sm btn-primary me-2"),
                ui.input_action_button("browser_cancel", "Cancel",
                                       class_="btn btn-secondary"),
            ),
            easy_close=True,
        )

    @reactive.effect
    @reactive.event(input.browse_exp_btn)
    def _open_browser_exp():
        browser_target.set("dir")
        start = input.exp_dir().strip()
        default_root = DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.is_dir() else Path.home()
        if start:
            p = Path(start)
            browser_cwd.set(p if p.is_dir() else default_root)
        else:
            browser_cwd.set(default_root)
        ui.modal_show(_make_browser_modal())

    @reactive.effect
    @reactive.event(input.browser_cancel)
    def _close_browser():
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.browser_up)
    def _browser_up():
        browser_cwd.set(browser_cwd().parent)
        ui.modal_show(_make_browser_modal())

    @reactive.effect
    @reactive.event(input.browser_open)
    def _browser_open():
        selected = input.browser_items()
        if not selected:
            return
        p = Path(selected)
        if p.is_dir():
            browser_cwd.set(p)
            ui.modal_show(_make_browser_modal())

    @reactive.effect
    @reactive.event(input.browser_select)
    def _browser_select():
        selected = input.browser_items()
        if not selected:
            return
        p = Path(selected)
        ui.update_text("exp_dir", value=str(p if p.is_dir() else browser_cwd()))
        ui.modal_remove()

    found_csvs: reactive.Value[list[Path]] = reactive.Value([])

    @reactive.effect
    @reactive.event(input.scan_btn)
    def _scan_for_csvs():
        exp = input.exp_dir().strip()
        if not exp:
            ui.notification_show("Enter an experiment directory first.", type="warning", duration=4)
            return
        csvs = find_combined_csvs(exp)
        if not csvs:
            ui.notification_show(
                "No *_combined.csv files found under that directory.", type="warning", duration=4
            )
        found_csvs.set(csvs)

    @render.ui
    def csv_picker_ui():
        csvs = found_csvs()
        if not csvs:
            return ui.div()
        exp_root = Path(input.exp_dir().strip()) if input.exp_dir().strip() else None

        def _csv_label(p: Path) -> str:
            if exp_root is not None:
                try:
                    rel = p.relative_to(exp_root)
                    return str(rel)
                except Exception:
                    pass
            return str(p)

        choices = {str(p): _csv_label(p) for p in csvs}
        return ui.div(
            ui.input_select("csv_path", "Select combined CSV", choices=choices, width="100%"),
            style="margin-top:6px;",
        )

    @reactive.effect
    @reactive.event(input.plot_type)
    def _update_thresh():
        defaults = {"strain_linear": 0.5, "strain_log": 0.5, "log2fc_scatter": 2.0}
        labels = {
            "strain_linear": "log₂(Y/X) hit threshold",
            "strain_log": "log₂(Y/X) hit threshold",
            "log2fc_scatter": "Δlog₂FC (Y − X) hit threshold",
        }
        pt = input.plot_type()
        if pt in defaults:
            ui.update_slider(
                "log2fc_hit_thresh",
                value=defaults[pt],
                label=labels.get(pt),
            )

    @reactive.effect
    @reactive.event(input.load_btn)
    def _load():
        path = input.csv_path().strip()
        if not path:
            return
        try:
            df = pl.read_csv(path, infer_schema_length=10000)
            raw_df.set(df)
            fcol = file_col(df)
            if "is_flagged" in df.columns and fcol and fcol in df.columns:
                flagged_rows = df.filter(
                    pl.col("is_flagged")
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .is_in(["true", "1", "yes", "y", "t"])
                )
                flagged_mask_paths.set(
                    set(
                        flagged_rows.select(pl.col(fcol).cast(pl.Utf8))
                        .drop_nulls()
                        .unique()
                        .to_series()
                        .to_list()
                    )
                )
            else:
                flagged_mask_paths.set(set())

            # Skip acquisition dates mis-labeled as day in older CSVs.
            days = meaningful_screen_days(df)
            if days:
                ui.update_checkbox_group(
                    "day_filter",
                    choices={str(d): f"Day {d}" for d in days},
                    selected=[str(d) for d in days],
                )

            # Strains
            if "strain" in df.columns:
                strains = sorted(df["strain"].cast(pl.Utf8).unique().to_list())
                strain_choices = {s: s for s in strains}
                ui.update_select("strain_a", choices=strain_choices,
                                 selected=strains[0] if strains else "")
                ui.update_select("strain_b", choices=strain_choices,
                                 selected=strains[1] if len(strains) > 1 else strains[0] if strains else "")
                ui.update_select("volcano_strain", choices=strain_choices,
                                 selected=strains[0] if strains else "")

            # Day selector only when a real developmental day exists.
            cmp_days = meaningful_screen_days(df)
            if cmp_days:
                ui.update_select(
                    "cmp_day",
                    choices={str(d): f"Day {d}" for d in cmp_days},
                    selected=str(cmp_days[0]),
                )
            else:
                ui.update_select("cmp_day", choices={}, selected="")

            inferred_exp = infer_experiment_dir_from_df(df)
            if inferred_exp:
                ui.update_text("exp_dir", value=inferred_exp)

        except Exception as e:
            ui.notification_show(f"Error loading CSV: {e}", type="error", duration=6)

    def _flagged_df_with_column():
        df = raw_df()
        if df is None:
            return None
        fcol = file_col(df)
        if fcol is None or fcol not in df.columns:
            return None
        flagged = list(flagged_mask_paths())
        base = df.drop("is_flagged") if "is_flagged" in df.columns else df
        return base.with_columns(
            pl.col(fcol).cast(pl.Utf8).is_in(flagged).alias("is_flagged")
        )

    def _persist_flags_to_loaded_csv():
        flagged_df = _flagged_df_with_column()
        if flagged_df is None:
            return None
        try:
            csv_path = input.csv_path().strip()
        except Exception:
            csv_path = ""
        if not csv_path:
            return None
        src = Path(csv_path)
        try:
            backup = src.with_name(f"{src.name}.backup")
            if src.exists() and not backup.exists():
                backup.write_bytes(src.read_bytes())
        except Exception:
            pass
        flagged_df.write_csv(str(src))
        raw_df.set(flagged_df)
        return src

    def _selected_flagged_paths_from_inputs() -> tuple[set[str], set[str]]:
        mapping = flag_input_map()
        selected: set[str] = set()
        for inp_id, mask_path in mapping.items():
            is_checked = False
            reader = getattr(input, inp_id, None)
            if callable(reader):
                try:
                    is_checked = bool(reader())
                except Exception:
                    is_checked = False
            if is_checked:
                selected.add(str(mask_path))
        displayed_paths = {str(v) for v in mapping.values() if v}
        return selected, displayed_paths

    def _apply_flags_for_current_view(show_notification: bool = True):
        mapping = flag_input_map()
        if not mapping:
            return
        selected, displayed_paths = _selected_flagged_paths_from_inputs()
        cur = set(flagged_mask_paths())
        cur -= displayed_paths
        cur |= selected
        if cur == flagged_mask_paths():
            _last_flag_selection.set(selected)
            return
        flagged_mask_paths.set(cur)
        _last_flag_selection.set(selected)
        saved_to = _persist_flags_to_loaded_csv()
        if show_notification:
            save_msg = f" Saved in: {saved_to.name}" if saved_to is not None else ""
            ui.notification_show(
                f"Updated flags: {len(selected)} flagged in current view ({len(cur)} total).{save_msg}",
                type="message",
                duration=3,
            )

    @reactive.effect
    def _auto_apply_flags_on_toggle():
        # Poll only while image cards are visible.
        if clicked_row() is None:
            return
        reactive.invalidate_later(0.7)
        mapping = flag_input_map()
        if not mapping:
            _last_flag_selection.set(set())
            return
        selected, _displayed = _selected_flagged_paths_from_inputs()
        if selected != _last_flag_selection():
            _apply_flags_for_current_view(show_notification=False)

    @reactive.effect
    @reactive.event(input.clear_all_flags_btn)
    def _clear_all_flags():
        for inp_id in list(flag_input_map().keys()):
            try:
                ui.update_checkbox(inp_id, value=False)
            except Exception:
                pass
        flagged_mask_paths.set(set())
        _last_flag_selection.set(set())
        saved_to = _persist_flags_to_loaded_csv()
        save_msg = f" Saved in: {saved_to.name}" if saved_to is not None else ""
        ui.notification_show(f"Cleared all flagged images.{save_msg}", type="message", duration=3)

    @render.ui
    def norm_ui():
        df = raw_df()
        if df is None:
            return ui.p("Load a CSV to configure normalization.",
                        style="color:#aaa; font-size:0.85em;")
        cols = metric_col_choices(df)
        metric_guess = default_metric_col(df) or (cols[0] if cols else "")
        return ui.div(
            ui.input_select("metric_col", "Metric column (body area / pharynx, µm²)",
                            choices=cols, selected=metric_guess, width="100%"),
            ui.p("Prefer gene-level columns ending in _gene or _qc_gene. "
                 "log₂FC baseline: EV if present, else empty-vector controls "
                 "(empty / empty_vector / L4440), else all controls.",
                 style="font-size:0.75em; color:#888; margin-top:4px;"),
        )

    @render.ui
    def day_filter_ui():
        df = raw_df()
        if df is None:
            return ui.p("Load a CSV first.", style="color:#aaa; font-size:0.85em;")
        days = meaningful_screen_days(df)
        if not days:
            return ui.p(
                "No developmental day in this screen (single timepoint).",
                style="color:#aaa; font-size:0.85em;",
            )
        return ui.input_checkbox_group(
            "day_filter", None,
            choices={str(d): f"Day {d}" for d in days},
            selected=[str(d) for d in days],
        )

    @render.ui
    def library_filter_ui():
        df = raw_df()
        if df is None or "lib_code" not in df.columns:
            return ui.p("Library filter available after loading CSV.", style="color:#aaa; font-size:0.85em;")
        libs = (
            df.with_columns(library_group_expr("lib_code").alias("__lib_group"))
            .select("__lib_group")
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        libs = [x for x in sorted(libs) if x in ("numeric", "ghr", "other")]
        if not libs:
            return ui.p("No libraries detected in lib_code.", style="color:#aaa; font-size:0.85em;")
        label_map = {
            "numeric": "Numeric@well library",
            "ghr": "GHR library",
            "other": "Other library codes",
        }
        choices = {"__all__": "All libraries (separate stats)"} | {lb: label_map.get(lb, lb) for lb in libs}
        return ui.input_select(
            "analysis_library",
            None,
            choices=choices,
            selected="__all__",
            width="100%",
        )

    @reactive.calc
    def filtered_df():
        df = raw_df()
        if df is None:
            return None
        # Exclude flagged images before any downstream stats.
        fcol = file_col(df)
        flagged = flagged_mask_paths()
        if fcol and flagged:
            df = df.filter(~pl.col(fcol).cast(pl.Utf8).is_in(list(flagged)))
        selected_days = None
        if df_has_meaningful_day(df) and hasattr(input, "day_filter"):
            selected_days = list(input.day_filter())
        if selected_days is not None and "day" in df.columns:
            df = df.filter(pl.col("day").cast(pl.Utf8).is_in(selected_days))
        try:
            lib_sel = input.analysis_library()
        except Exception:
            lib_sel = "__all__"
        df = apply_library_filter(df, lib_sel)
        return df

    @reactive.calc
    def normed_df():
        """Add metric_um2_per_ph and log2fc_vs_ev; baseline omits plate."""
        df = raw_df()
        if df is None:
            return None
        fcol = file_col(df)
        flagged = flagged_mask_paths()
        if fcol and flagged:
            df = df.filter(~pl.col(fcol).cast(pl.Utf8).is_in(list(flagged)))
        try:
            lib_sel = input.analysis_library()
        except Exception:
            lib_sel = "__all__"
        df = apply_library_filter(df, lib_sel)

        try:
            metric_col = input.metric_col()
        except Exception:
            return df  

        metric_col = selected_metric_col(df, metric_col)
        if not metric_col:
            return df

        df = collapse_to_gene_level(df, metric_col)
        df = df.with_columns(pl.col(metric_col).alias("metric_um2_per_ph"))

        # Control baseline per strain (+ day); omit plate for split-plate screens.
        ev_group_keys = screen_group_keys(df, "strain", "day")
        if not ev_group_keys:
            return df

        controls = select_normalization_reference_df(df)
        ev_df = (
            controls
            .group_by(ev_group_keys, maintain_order=True)
            .agg(pl.col("metric_um2_per_ph").median().alias("ev_metric_um2_per_ph"))
        )
        df = df.join(ev_df, on=ev_group_keys, how="left")

        fallback_keys = screen_group_keys(df, "strain")
        if fallback_keys and fallback_keys != ev_group_keys:
            ev_fb = (
                controls
                .group_by(fallback_keys, maintain_order=True)
                .agg(pl.col("metric_um2_per_ph").median().alias("__ev_metric_fallback"))
            )
            df = (
                df.join(ev_fb, on=fallback_keys, how="left")
                .with_columns(
                    pl.coalesce(
                        [pl.col("ev_metric_um2_per_ph"), pl.col("__ev_metric_fallback")]
                    ).alias("ev_metric_um2_per_ph")
                )
                .drop("__ev_metric_fallback")
            )

        df = df.with_columns(
            pl.when(
                (pl.col("metric_um2_per_ph") > 0)
                & (pl.col("ev_metric_um2_per_ph") > 0)
            )
            .then(
                (pl.col("metric_um2_per_ph") / pl.col("ev_metric_um2_per_ph")).log()
                / pl.lit(np.log(2.0))
            )
            .otherwise(None)
            .alias("log2fc_vs_ev")
        )

        return df

    @reactive.calc
    def well_df():
        df = filtered_df()
        if df is None:
            return None
        key = [c for c in ["strain", "day", "plate", "well96", "gene_name", "lib_code", "is_control"]
               if c in df.columns]
        num = numeric_cols(df)
        return df.group_by(key, maintain_order=True).agg(
            [pl.col(c).mean().alias(c) for c in num]
        )

    @reactive.calc
    def comparison_df():
        """Strain A vs B metrics joined on gene keys (median across plates)."""
        ndf = normed_df()
        if ndf is None:
            return None

        if "metric_um2_per_ph" not in ndf.columns:
            return None

        try:
            plot_type = input.plot_type()
        except Exception:
            return None
        if plot_type not in ("strain_linear", "strain_log", "log2fc_scatter"):
            return None

        try:
            strain_a = str(input.strain_a())
            strain_b = str(input.strain_b())
            cmp_day = str(input.cmp_day())
        except Exception:
            return None

        if not strain_a or not strain_b or "strain" not in ndf.columns:
            return None

        if cmp_day and df_has_meaningful_day(ndf) and "day" in ndf.columns:
            ndf = ndf.filter(pl.col("day").cast(pl.Utf8) == cmp_day)

        metric = "log2fc_vs_ev" if plot_type == "log2fc_scatter" else "metric_um2_per_ph"
        if metric not in ndf.columns:
            return None

        # Cast join keys to avoid dtype mismatches dropping genes.
        ndf = ndf.with_columns(
            pl.col("gene_name").cast(pl.Utf8).alias("gene_name"),
            pl.col("lib_code").cast(pl.Utf8).alias("lib_code"),
            pl.col("is_control").cast(pl.Utf8).str.to_lowercase().alias("is_control"),
        )
        ndf = prepare_comparison_metrics_df(ndf)
        group_keys = [
            c
            for c in ["gene_name", "lib_code", "is_control"]
            if c in ndf.columns
        ]
        if not group_keys:
            group_keys = [c for c in ["gene_name", "is_control"] if c in ndf.columns]

        def strain_metric(strain_id: str) -> pl.DataFrame:
            sub = ndf.filter(pl.col("strain").cast(pl.Utf8) == strain_id)
            return (
                sub.group_by(group_keys, maintain_order=True)
                .agg(pl.col(metric).median().alias(f"m_{strain_id}"))
            )

        df_a = strain_metric(strain_a)
        df_b = strain_metric(strain_b)
        col_a, col_b = f"m_{strain_a}", f"m_{strain_b}"

        merged = df_a.join(df_b, on=group_keys, how="inner")

        thresh = input.log2fc_hit_thresh()
        try:
            hit_direction = str(input.hit_direction())
        except Exception:
            hit_direction = "symmetric"
        if plot_type == "log2fc_scatter":
            merged = merged.with_columns(
                pl.when(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
                .then(pl.col(col_b) - pl.col(col_a))
                .otherwise(None)
                .alias("_log2_ratio")
            )
        else:
            eps = 1e-9
            merged = merged.with_columns(
                pl.when(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
                .then(
                    (
                        (pl.col(col_b) + pl.lit(eps))
                        / (pl.col(col_a) + pl.lit(eps))
                    ).log()
                    / pl.lit(np.log(2.0))
                )
                .otherwise(None)
                .alias("_log2_ratio")
            )

        if plot_type == "log2fc_scatter" and hit_direction == "neg":
            hit_expr = pl.col("_log2_ratio") <= -thresh
        elif plot_type == "log2fc_scatter" and hit_direction == "pos":
            hit_expr = pl.col("_log2_ratio") >= thresh
        else:
            hit_expr = pl.col("_log2_ratio").abs() >= thresh

        is_ctrl_col = "is_control" if "is_control" in merged.columns else None
        if is_ctrl_col:
            merged = merged.with_columns(
                pl.when(pl.col(is_ctrl_col).cast(pl.Utf8).str.to_lowercase() == "true")
                .then(pl.lit("control"))
                .when(hit_expr)
                .then(pl.lit("hit"))
                .otherwise(pl.lit("non-hit"))
                .alias("label")
            )
        else:
            merged = merged.with_columns(
                pl.when(hit_expr)
                .then(pl.lit("hit"))
                .otherwise(pl.lit("non-hit"))
                .alias("label")
            )

        merged = merged.with_columns(plot_color_expr().alias("plot_color"))

        return merged, col_a, col_b

    @reactive.calc
    def volcano_df():
        df = raw_df()
        if df is None:
            return None
        fcol = file_col(df)
        flagged = flagged_mask_paths()
        if fcol and flagged:
            df = df.filter(~pl.col(fcol).cast(pl.Utf8).is_in(list(flagged)))
        try:
            lib_sel = input.analysis_library()
        except Exception:
            lib_sel = "__all__"
        df = apply_library_filter(df, lib_sel)
        selected_days = None
        if df_has_meaningful_day(df) and hasattr(input, "day_filter"):
            selected_days = list(input.day_filter())
        if selected_days is not None and "day" in df.columns:
            df = df.filter(pl.col("day").cast(pl.Utf8).is_in(selected_days))

        try:
            metric = input.metric_col()
        except Exception:
            return None
        metric = selected_metric_col(df, metric)
        if not metric:
            return None
        wdf = collapse_to_gene_level(df, metric)

        pval_thresh = input.pval_threshold()

        try:
            volcano_mode = input.volcano_mode()
        except Exception:
            volcano_mode = "combined"

        if volcano_mode == "per_strain" and "strain" in wdf.columns:
            try:
                selected_strain = str(input.volcano_strain())
            except Exception:
                selected_strain = None
            if selected_strain:
                wdf = wdf.filter(pl.col("strain").cast(pl.Utf8) == selected_strain)

        rows = []
        if "lib_code" in wdf.columns:
            wdf = wdf.with_columns(library_group_expr("lib_code").alias("__lib_group"))
            libs = (
                wdf["__lib_group"].drop_nulls().unique().to_list()
            )
            libs = [x for x in sorted(libs) if x in ("numeric", "ghr", "other")]
        else:
            libs = [None]

        for lib in libs:
            lib_df = (
                wdf
                if lib is None
                else wdf.filter(pl.col("__lib_group").is_in([str(lib), "control"]))
            )
            # Volcano baseline: EV, else empty-vector controls, else all controls.
            ref_df = select_normalization_reference_df(lib_df)
            ev_vals = ref_df[metric].drop_nulls().to_numpy()
            if len(ev_vals) < 2:
                continue
            ev_median = float(np.median(ev_vals))

            for gene, grp in lib_df.group_by("gene_name"):
                gene_name = gene[0] if isinstance(gene, tuple) else gene
                is_ctrl = grp["is_control"][0]
                gene_vals = grp[metric].drop_nulls().to_numpy()
                if len(gene_vals) < 1:
                    continue

                gene_mean = float(np.mean(gene_vals))
                if gene_mean <= 0 or ev_median <= 0:
                    fold_change = float("nan")
                else:
                    fold_change = float(np.log2(gene_mean / ev_median))

                if len(gene_vals) >= 2 and len(ev_vals) >= 2:
                    _, pval = stats.ttest_ind(gene_vals, ev_vals, equal_var=False)
                else:
                    pval = 1.0

                rows.append({
                    "gene_name": gene_name,
                    "lib_code": grp["lib_code"][0] if "lib_code" in grp.columns else (lib or ""),
                    "library_group": str(lib) if lib is not None else "",
                    "is_control": is_ctrl,
                    "log2_fold_change": float(fold_change),
                    "neg_log10_pval": float(-np.log10(max(pval, 1e-300))),
                    "pval": float(pval),
                    metric: float(gene_mean),
                })

        if not rows:
            return None

        vdf = pl.DataFrame(rows)
        try:
            fc_thresh = float(input.volcano_log2fc_thresh())
        except Exception:
            fc_thresh = 1.0

        return (
            vdf.with_columns(
                pl.when(pl.col("is_control").cast(pl.Utf8).str.to_lowercase() == "true")
                .then(pl.lit("control"))
                .when(
                    (pl.col("log2_fold_change").abs() >= fc_thresh)
                    & (pl.col("pval") <= pval_thresh)
                )
                .then(pl.lit("hit"))
                .otherwise(pl.lit("non-hit"))
                .alias("label")
            ).with_columns(plot_color_expr().alias("plot_color"))
        )

    def apply_vis_filter(df: pl.DataFrame) -> pl.DataFrame:
        mask = pl.lit(False)
        if input.show_hits():
            mask = mask | (pl.col("label") == "hit")
        if input.show_nonhits():
            mask = mask | (pl.col("label") == "non-hit")
        if input.show_controls():
            mask = mask | (pl.col("label") == "control")
        return df.filter(mask)

    @render_widget
    def main_plot():
        color_map = PLOT_COLOR_MAP
        plot_type = input.plot_type()

        #  Volcano                                                             #
        if plot_type == "volcano":
            vdf = volcano_df()
            if vdf is None:
                return empty_fig("Load a CSV and select a metric")

            plot_df = apply_vis_filter(vdf).with_row_index("__plot_idx")
            _plot_df_cache.set(plot_df)

            try:
                v_mode = input.volcano_mode()
                v_strain = str(input.volcano_strain()) if v_mode == "per_strain" else None
            except Exception:
                v_mode, v_strain = "combined", None
            volcano_title = (
                f"Volcano | Strain {v_strain}" if v_mode == "per_strain" and v_strain
                else "Volcano | All strains combined"
            )

            pdf = plot_df.to_pandas()
            pdf = sanitize_xy_pdf(pdf, "log2_fold_change", "neg_log10_pval")
            if pdf.empty:
                return empty_fig("No finite data points available for this selection")

            x_col, y_col = "log2_fold_change", "neg_log10_pval"
            hover_cols = ["gene_name", "lib_code", "pval", "label", "plot_color"]
            hover_labels = {
                x_col: "log₂ fold change vs EV",
                y_col: "-log₁₀(p-value)",
                "gene_name": "gene_name",
                "lib_code": "lib_code",
                "pval": "pval",
                "label": "Hit call",
                "plot_color": "Category",
            }
            fig = build_layered_scatter(
                pdf,
                x_col,
                y_col,
                color_col="plot_color",
                color_map=color_map,
                color_order=PLOT_COLOR_ORDER,
                title=volcano_title,
                xaxis_title="log₂ fold change vs EV",
                yaxis_title="-log₁₀(p-value)",
                hover_cols=hover_cols,
                hover_labels=hover_labels,
            )
            _plot_axes_cache.set((x_col, y_col))
            pval_line = -np.log10(input.pval_threshold())
            fig.add_hline(y=pval_line, line_dash="dash", line_color="#888",
                          annotation_text=f"p={input.pval_threshold()}")
            try:
                fc_abs = float(input.volcano_log2fc_thresh())
            except Exception:
                fc_abs = 1.0
            if fc_abs > 0:
                fig.add_vline(
                    x=fc_abs, line_dash="dash", line_color="#aaa",
                    annotation_text=f"|log₂FC|={fc_abs}",
                    annotation_position="top",
                )
                fig.add_vline(x=-fc_abs, line_dash="dash", line_color="#aaa")

        #  Strain comparison — linear or log scale                            #
        elif plot_type in ("strain_linear", "strain_log"):
            result = comparison_df()
            if result is None:
                return empty_fig("Load a CSV, configure normalization, and select strains")

            cdf, col_a, col_b = result
            plot_df = apply_vis_filter(cdf).with_row_index("__plot_idx")
            _plot_df_cache.set(plot_df)

            strain_a = input.strain_a()
            strain_b = input.strain_b()
            cmp_days = meaningful_screen_days(raw_df())
            day_suffix = f", Day {cmp_days[0]}" if len(cmp_days) == 1 else ""
            label_x = f"{strain_a} average adult size (µm²/pharynx)"
            label_y = f"{strain_b} average adult size (µm²/pharynx)"

            hover_cols = [c for c in ["gene_name", "lib_code", "_log2_ratio", "label", "plot_color"]
                          if c in plot_df.columns]
            pdf = plot_df.to_pandas()
            pdf = sanitize_xy_pdf(pdf, col_a, col_b)
            if pdf.empty:
                return empty_fig("No finite data points available for this selection")

            hover_labels = {
                col_a: label_x,
                col_b: label_y,
                "_log2_ratio": f"log₂({strain_b}/{strain_a}) = log₂(Y/X)",
                "gene_name": "gene_name",
                "lib_code": "lib_code",
                "label": "Hit call",
                "plot_color": "Category",
            }
            fig = build_layered_scatter(
                pdf,
                col_a,
                col_b,
                color_col="plot_color",
                color_map=color_map,
                color_order=PLOT_COLOR_ORDER,
                title=f"Strain {strain_b} vs Strain {strain_a}{day_suffix}",
                xaxis_title=label_x,
                yaxis_title=label_y,
                hover_cols=hover_cols,
                hover_labels=hover_labels,
            )
            _plot_axes_cache.set((col_a, col_b))

            if plot_type == "strain_log":
                min_nonzero = pdf[[col_a, col_b]].replace(0, float("nan")).min().min()
                min_nonzero = max(min_nonzero, 1.0) if not np.isnan(min_nonzero) else 1.0
                data_max = pdf[[col_a, col_b]].max().max()
                if not np.isfinite(data_max) or data_max <= 0:
                    data_max = min_nonzero
                log_axis = dict(
                    type="log",
                    range=[np.log10(min_nonzero), np.log10(data_max * 1.1)],
                    dtick=1,
                    exponentformat="power",
                    showexponent="all",
                )
                fig.update_xaxes(**log_axis)
                fig.update_yaxes(**log_axis)
                fig.add_shape(type="line",
                              x0=min_nonzero, y0=min_nonzero,
                              x1=data_max, y1=data_max,
                              xref="x", yref="y",
                              line=dict(color="gray", width=1))
            else:
                data_max = pdf[[col_a, col_b]].max().max() * 1.05
                fig.add_shape(type="line", x0=0, y0=0, x1=data_max, y1=data_max,
                              line=dict(color="gray", width=1))

        #  log₂FC scatter (strain pair)                                        #
        elif plot_type == "log2fc_scatter":
            result = comparison_df()
            if result is None:
                return empty_fig("Load a CSV, configure normalization, and select strains")

            cdf, col_a, col_b = result
            plot_df = apply_vis_filter(cdf).with_row_index("__plot_idx")
            _plot_df_cache.set(plot_df)

            strain_a = input.strain_a()
            strain_b = input.strain_b()
            cmp_days = meaningful_screen_days(raw_df())
            day_suffix = f", Day {cmp_days[0]}" if len(cmp_days) == 1 else ""
            label_x = f"{strain_a} log2FC vs EV"
            label_y = f"{strain_b} log2FC vs EV"

            hover_cols = [c for c in ["gene_name", "lib_code", "_log2_ratio", "label", "plot_color"]
                          if c in plot_df.columns]

            # Clip axes to notebook-style ±5.
            MAX_ABS = 5.0
            pdf = plot_df.to_pandas()
            pdf = sanitize_xy_pdf(pdf, col_a, col_b)
            if pdf.empty:
                return empty_fig("No finite data points available for this selection")
            pdf[col_a] = pdf[col_a].clip(-MAX_ABS, MAX_ABS)
            pdf[col_b] = pdf[col_b].clip(-MAX_ABS, MAX_ABS)

            hover_labels = {
                col_a: label_x,
                col_b: label_y,
                "_log2_ratio": f"Δlog₂FC ({strain_b} − {strain_a}) = Y − X",
                "gene_name": "gene_name",
                "lib_code": "lib_code",
                "label": "Hit call",
                "plot_color": "Category",
            }
            fig = build_layered_scatter(
                pdf,
                col_a,
                col_b,
                color_col="plot_color",
                color_map=color_map,
                color_order=PLOT_COLOR_ORDER,
                title=f"Strain {strain_b} vs Strain {strain_a}{day_suffix}",
                xaxis_title=label_x,
                yaxis_title=label_y,
                hover_cols=hover_cols,
                hover_labels=hover_labels,
            )
            _plot_axes_cache.set((col_a, col_b))

            fig.add_shape(type="line", x0=-MAX_ABS, y0=-MAX_ABS, x1=MAX_ABS, y1=MAX_ABS,
                          line=dict(color="gray", width=1))

            fig.add_hline(y=0, line_color="#ccc", line_width=0.8)
            fig.add_vline(x=0, line_color="#ccc", line_width=0.8)
            fig.update_xaxes(range=[-MAX_ABS, MAX_ABS])
            fig.update_yaxes(range=[-MAX_ABS, MAX_ABS])

        def on_click(trace, points, selector):
            if not points.xs:
                return
            cached = _plot_df_cache()
            xcol, ycol = _plot_axes_cache()
            if cached is None or not xcol or not ycol:
                return
            row = pick_plot_row_at_click(
                cached, points.xs[0], points.ys[0], xcol, ycol
            )
            if row is not None:
                clicked_row.set(row)

        for trace in fig.data:
            trace.on_click(on_click)

        fig = sanitize_figure_for_json(fig)
        _plot_fig_cache.set(fig)
        return fig

    @render.ui
    def cache_status_ui():
        _ = preload_started()
        with _cache_lock:
            done = _cache_stats["done"]
            total = _cache_stats["total"]
            priority_done = _cache_stats["priority_done"]
            priority_total = _cache_stats["priority_total"]
            priority_complete = bool(_cache_stats["priority_complete"])

        any_running = any(t.is_alive() for t in _preload_threads)
        if (total > 0 and done < total) or any_running:
            reactive.invalidate_later(1)

        if total == 0:
            return ui.div()

        pct = int(100 * done / total)
        color = "#2ecc71" if done >= total else "#3498db"
        label = f"Images cached: {done}/{total} ({pct}%)"
        priority_label = None
        if priority_total > 0:
            if priority_complete:
                priority_label = "Hit/control priority phase complete."
            else:
                ppct = int(100 * priority_done / priority_total)
                priority_label = (
                    f"Priority (hits+controls): {priority_done}/{priority_total} ({ppct}%)"
                )
        backend_note = (
            f"Preview loader: {image_backend_name()} "
            f"(max width {PREVIEW_MAX_WIDTH}px)"
        )
        return ui.div(
            ui.p(label, style=f"font-size:0.78em; color:{color}; margin:0 0 2px 0;"),
            ui.p(
                priority_label,
                style="font-size:0.74em; color:#666; margin:0 0 3px 0;",
            ) if priority_label else ui.div(),
            ui.p(
                backend_note,
                style="font-size:0.72em; color:#888; margin:0 0 3px 0;",
            ),
            ui.div(style=(
                f"height:4px; width:{pct}%; background:{color}; "
                "border-radius:2px; transition:width 0.5s;"
            )),
            style="margin:2px 0 6px 0;",
        )

    def _manual_preload_impl(*, hits_controls_only: bool) -> None:
        """Preload images for current plot selection in background."""
        plot_df = _plot_df_cache()
        if plot_df is None:
            ui.notification_show("No plot data to preload yet.", type="warning", duration=3)
            return
        if hits_controls_only:
            if "label" not in plot_df.columns:
                ui.notification_show(
                    "This plot has no hit/control labels; use 'Preload all plot images'.",
                    type="warning",
                    duration=4,
                )
                return
            plot_df = plot_df.filter(
                pl.col("label").cast(pl.Utf8).str.to_lowercase().is_in(["hit", "control"])
            )
            if plot_df.height == 0:
                ui.notification_show(
                    "No visible hits or controls (check Show: Hits / Controls).",
                    type="warning",
                    duration=4,
                )
                return
        try:
            pt = input.plot_type()
        except Exception:
            pt = None

        # Preload needs per-image rows (gene-collapsed tables drop *_file columns).
        if pt in ("strain_linear", "strain_log", "log2fc_scatter"):
            df = raw_df()
            if df is not None:
                fcol_flag = file_col(df)
                flagged = flagged_mask_paths()
                if fcol_flag and flagged:
                    df = df.filter(~pl.col(fcol_flag).cast(pl.Utf8).is_in(list(flagged)))
                try:
                    lib_sel = input.analysis_library()
                except Exception:
                    lib_sel = "__all__"
                df = apply_library_filter(df, lib_sel)
        else:
            df = filtered_df()
        if df is None:
            ui.notification_show("No data available for preloading yet.", type="warning", duration=3)
            return
        exp_dir = input.exp_dir().strip()
        if not exp_dir:
            ui.notification_show("Set experiment directory before preloading images.", type="warning", duration=3)
            return
        ch = int(input.image_channel()) if hasattr(input, "image_channel") else 1
        fcol_name = display_file_col(df, ch)
        if fcol_name is None or fcol_name not in df.columns:
            ui.notification_show("No image file column found for preloading.", type="warning", duration=3)
            return

        if any(t.is_alive() for t in _preload_threads):
            ui.notification_show("Preload already running in background.", type="message", duration=3)
            return

        source = df
        label_priority: dict[str, int] = {}
        key_cols = preload_join_key_cols(plot_df, df)
        if set(["gene_name", "label"]).issubset(set(plot_df.columns)):
            try:
                if key_cols:
                    prio_df = (
                        plot_df.select(
                            *[pl.col(c).cast(pl.Utf8).alias(c) for c in key_cols],
                            pl.col("label").cast(pl.Utf8).str.to_lowercase().alias("label"),
                        )
                        .drop_nulls()
                        .unique()
                        .with_columns(
                            pl.when(pl.col("label") == "hit")
                            .then(pl.lit(0))
                            .when(pl.col("label") == "control")
                            .then(pl.lit(1))
                            .otherwise(pl.lit(2))
                            .alias("__prio"),
                            pl.concat_str([pl.col(c) for c in key_cols], separator="|").alias("__k"),
                        )
                        .group_by("__k", maintain_order=True)
                        .agg(pl.col("__prio").min().alias("__prio"))
                    )
                    label_priority = {str(r["__k"]): int(r["__prio"]) for r in prio_df.iter_rows(named=True)}
                else:
                    prio_df = (
                        plot_df.select(
                            pl.col("gene_name").cast(pl.Utf8).alias("gene_name"),
                            pl.col("label").cast(pl.Utf8).str.to_lowercase().alias("label"),
                        )
                        .drop_nulls()
                        .unique()
                        .with_columns(
                            pl.when(pl.col("label") == "hit")
                            .then(pl.lit(0))
                            .when(pl.col("label") == "control")
                            .then(pl.lit(1))
                            .otherwise(pl.lit(2))
                            .alias("__prio")
                        )
                        .group_by("gene_name", maintain_order=True)
                        .agg(pl.col("__prio").min().alias("__prio"))
                    )
                    label_priority = {
                        str(r["gene_name"]): int(r["__prio"])
                        for r in prio_df.iter_rows(named=True)
                    }
            except Exception:
                label_priority = {}

        if key_cols:
            plotted_keys = (
                plot_df.select(*[pl.col(c).cast(pl.Utf8).alias(c) for c in key_cols])
                .drop_nulls()
                .unique()
            )
            if plotted_keys.height > 0:
                source = (
                    source.with_columns(*[pl.col(c).cast(pl.Utf8).alias(c) for c in key_cols])
                    .join(plotted_keys, on=key_cols, how="inner")
                )
        elif "gene_name" in plot_df.columns and "gene_name" in df.columns:
            genes = (
                plot_df.select(pl.col("gene_name").cast(pl.Utf8))
                .drop_nulls()
                .unique()
                .to_series()
                .to_list()
            )
            if genes:
                source = source.filter(pl.col("gene_name").cast(pl.Utf8).is_in(genes))

        # Limit preload to strains shown on the current plot.
        if "strain" in source.columns and "strain" in plot_df.columns:
            try:
                plotted_strains = (
                    plot_df.select(pl.col("strain").cast(pl.Utf8))
                    .drop_nulls()
                    .unique()
                    .to_series()
                    .to_list()
                )
            except Exception:
                plotted_strains = []
            if plotted_strains:
                source = source.filter(pl.col("strain").cast(pl.Utf8).is_in(plotted_strains))

        if pt in ("strain_linear", "strain_log", "log2fc_scatter"):
            try:
                cmp_day = str(input.cmp_day())
            except Exception:
                cmp_day = None
            if cmp_day and df_has_meaningful_day(source) and "day" in source.columns:
                source = source.filter(pl.col("day").cast(pl.Utf8) == cmp_day)
            if "strain" in source.columns:
                try:
                    sa = str(input.strain_a())
                    sb = str(input.strain_b())
                    source = source.filter(pl.col("strain").cast(pl.Utf8).is_in([sa, sb]))
                except Exception:
                    pass
        else:
            if "day" in source.columns and df_has_meaningful_day(source):
                try:
                    selected_days = list(input.day_filter()) if hasattr(input, "day_filter") else []
                except Exception:
                    selected_days = []
                if selected_days:
                    source = source.filter(pl.col("day").cast(pl.Utf8).is_in([str(d) for d in selected_days]))

        select_exprs = [pl.col(fcol_name).cast(pl.Utf8).alias("_fcol_")]
        obj_col_name = (
            f"{fcol_name[:-5]}_object_csv" if str(fcol_name).endswith("_file") else None
        )
        if obj_col_name and obj_col_name in source.columns:
            select_exprs.append(pl.col(obj_col_name).cast(pl.Utf8).alias("_obj_csv"))
        for c in key_cols:
            if c == "is_control":
                select_exprs.append(pl.col(c).cast(pl.Utf8).str.to_lowercase().alias(f"__k_{c}"))
            else:
                select_exprs.append(pl.col(c).cast(pl.Utf8).alias(f"__k_{c}"))
        rows_df = (
            source.select(*select_exprs)
            .drop_nulls(subset=["_fcol_"])
        )
        # Cap images per plotted point on comparison plots.
        if pt in ("strain_linear", "strain_log", "log2fc_scatter") and key_cols:
            group_cols = [f"__k_{c}" for c in key_cols]
            order_cols = [c for c in ["__k_plate", "__k_well96", "__k_row384", "__k_col384", "_fcol_"] if c in rows_df.columns]
            if order_cols:
                rows_df = rows_df.sort(order_cols)
            rows_df = (
                rows_df.with_columns(
                    pl.col("_fcol_").cum_count().over(group_cols).alias("__per_point_idx")
                )
                .filter(pl.col("__per_point_idx") <= MAX_PRELOAD_PER_PLOTTED_POINT)
                .drop("__per_point_idx")
            )
        rows_df = rows_df.unique(subset=["_fcol_"])
        if label_priority:
            if key_cols:
                kcols = [f"__k_{c}" for c in key_cols]
                rows_df = rows_df.with_columns(
                    pl.concat_str([pl.col(c) for c in kcols], separator="|")
                    .replace_strict(label_priority, default=2)
                    .cast(pl.Int64)
                    .alias("__prio")
                ).sort(["__prio", "_fcol_"])
            else:
                rows_df = rows_df.with_columns(
                    pl.lit(2).cast(pl.Int64).alias("__prio")
                ).sort(["__prio", "_fcol_"])
        rows = rows_df.to_dicts()
        try:
            overlay_mode = str(input.overlay_mode()) if hasattr(input, "overlay_mode") else "none"
        except Exception:
            overlay_mode = "none"
        try:
            overlay_alpha = float(input.overlay_alpha()) if hasattr(input, "overlay_alpha") else 0.35
        except Exception:
            overlay_alpha = 0.35
        mask_mode = "eq1"

        def _variants_needed(r: dict) -> list[str]:
            obj = r.get("_obj_csv")
            obj_str = str(obj) if obj not in (None, "") else None
            return _preload_variants_for_row(
                r.get("_fcol_"),
                exp_dir,
                ch,
                overlay_mode=overlay_mode,
                overlay_alpha=overlay_alpha,
                object_csv_path=obj_str,
                mask_positive_mode=mask_mode,
            )

        to_load = [r for r in rows if _variants_needed(r)]
        n_variants = sum(len(_variants_needed(r)) for r in to_load)

        if not to_load:
            ui.notification_show("All images for current plot are already cached.", type="message", duration=3)
            return

        with _cache_lock:
            _cache_stats["done"] = 0
            _cache_stats["total"] = n_variants
            _cache_stats["priority_done"] = 0
            _cache_stats["priority_total"] = sum(
                len(_variants_needed(r))
                for r in to_load
                if int(r.get("__prio", 2)) <= 1
            )
            _cache_stats["priority_complete"] = (
                1 if _cache_stats["priority_total"] == 0 else 0
            )
        preload_started.set(preload_started() + 1)

        chunk_size = max(1, len(to_load) // N_PRELOAD_THREADS)
        chunks = [to_load[i:i + chunk_size] for i in range(0, len(to_load), chunk_size)]
        _preload_threads.clear()
        for chunk in chunks:
            t = threading.Thread(
                target=_preload_worker,
                args=(chunk, exp_dir, ch),
                kwargs={
                    "overlay_mode": overlay_mode,
                    "overlay_alpha": overlay_alpha,
                    "mask_positive_mode": mask_mode,
                },
                daemon=True,
            )
            t.start()
            _preload_threads.append(t)
        scope_note = "hits & controls only" if hits_controls_only else "all visible plot points"
        overlay_note = (
            f" + mask overlay ({overlay_mode})"
            if overlay_mode != "none"
            else ""
        )
        print(
            f"[preload] Started {len(chunks)} threads for {n_variants} previews "
            f"({len(to_load)} images{overlay_note}, {scope_note})",
            flush=True,
        )
        ui.notification_show(
            f"Preloading {n_variants} previews ({len(to_load)} images{overlay_note}, {scope_note})…",
            type="message",
            duration=4,
        )

    @reactive.effect
    @reactive.event(input.preload_hits_controls_btn)
    def _manual_preload_hits_controls():
        _manual_preload_impl(hits_controls_only=True)

    @reactive.effect
    @reactive.event(input.preload_all_plot_btn)
    def _manual_preload_all_plot():
        _manual_preload_impl(hits_controls_only=False)

    _DOWNLOAD_INTERNAL_COLS = frozenset({"__plot_idx", "plot_color"})

    def _safe_download_slug(value) -> str:
        s = str(value or "").strip()
        return re.sub(r"[^\w.\-]+", "_", s) if s else "unknown"

    def _comparison_export_stem() -> str | None:
        try:
            a = _safe_download_slug(input.strain_a())
            b = _safe_download_slug(input.strain_b())
            day = _safe_download_slug(input.cmp_day())
            return f"{b}vs{a}_day{day}"
        except Exception:
            return None

    def _export_stem_prefix() -> str:
        ptype = _safe_download_slug(input.plot_type())
        if ptype in ("strain_linear", "strain_log", "log2fc_scatter"):
            stem = _comparison_export_stem()
            if stem:
                return stem
        if ptype == "volcano":
            try:
                if input.volcano_mode() == "per_strain":
                    return _safe_download_slug(input.volcano_strain())
            except Exception:
                pass
            return "all_strains"
        return "export"

    def _export_filename_hits() -> str:
        return f"{_export_stem_prefix()}_hits_{_safe_download_slug(input.plot_type())}.csv"

    def _export_filename_plot_html() -> str:
        return f"{_export_stem_prefix()}_plot_{_safe_download_slug(input.plot_type())}.html"

    def _plot_df_for_export() -> pl.DataFrame | None:
        """Rebuild plot table at download time (cache can be stale after UI changes)."""
        plot_type = input.plot_type()
        try:
            if plot_type == "volcano":
                vdf = volcano_df()
                return apply_vis_filter(vdf) if vdf is not None else None
            if plot_type in ("strain_linear", "strain_log", "log2fc_scatter"):
                result = comparison_df()
                if result is None:
                    return None
                cdf, _, _ = result
                return apply_vis_filter(cdf)
        except Exception:
            pass
        return _plot_df_cache()

    def _csv_download_bytes(df: pl.DataFrame) -> bytes:
        return df.write_csv().encode("utf-8")

    @render.download(filename=_export_filename_plot_html, media_type="text/html")
    def dl_plot_html():
        import plotly.io as pio

        fig = _plot_fig_cache()
        if fig is None:
            msg = (
                "<html><body><p>No plot available. "
                "Load data and wait for the plot to finish rendering, then try again.</p></body></html>"
            )
            yield msg.encode("utf-8")
            return
        try:
            html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
            yield html.encode("utf-8") if isinstance(html, str) else html
        except Exception as exc:
            err = (
                f"<html><body><p>Plot export failed: {exc}</p></body></html>"
            )
            yield err.encode("utf-8")

    @render.download(filename=_export_filename_hits, media_type="text/csv")
    def dl_hits_csv():
        try:
            df = _plot_df_for_export()
            if df is None:
                note = pl.DataFrame(
                    {
                        "message": [
                            "No plot data. Load a CSV, set strains/day, "
                            "wait for the plot to update, then download again."
                        ]
                    }
                )
                yield _csv_download_bytes(note)
                return

            drop = [c for c in df.columns if c in _DOWNLOAD_INTERNAL_COLS]
            if drop:
                df = df.drop(drop)

            if "label" not in df.columns:
                note = pl.DataFrame({"message": ["Plot table has no hit labels."]})
                yield _csv_download_bytes(note)
                return

            hits = df.filter(pl.col("label") == "hit")
            if hits.height == 0:
                yield b"gene_name,message\n,(no hits with current thresholds)\n"
                return
            yield _csv_download_bytes(hits)
        except Exception as exc:
            err = pl.DataFrame({"error": [str(exc)]})
            yield _csv_download_bytes(err)

    @render.download(filename=lambda: Path(input.csv_path()).name if input.csv_path() else "combined_with_flags.csv")
    def dl_flagged_csv():
        flagged_df = _flagged_df_with_column()
        if flagged_df is None:
            yield b"message\n(no flagged data loaded)\n"
            return
        yield _csv_download_bytes(flagged_df)

    @render.ui
    def image_dl_picker():
        flagged_now = flagged_mask_paths()
        images = [
            t for t in _displayed_images()
            if str(t[2]) not in flagged_now
        ]
        if not images:
            return ui.div()
        choices = {
            "__all_tiff__": "All non-flagged images | TIFF (ZIP)",
            "__collage__": "Non-flagged collage (PNG)",
        }
        for i, (lbl, _data_uri, _mpath) in enumerate(images):
            choices[f"tiff_{i}"] = lbl
        return ui.div(
            ui.input_select("dl_image_choice", None,
                            choices=choices, selected="__all_tiff__", width="240px"),
            ui.download_button("dl_images", "⬇ Download",
                               class_="btn btn-sm btn-outline-secondary"),
        )

    def _dl_filename():
        flagged_now = flagged_mask_paths()
        images = [t for t in _displayed_images() if str(t[2]) not in flagged_now]
        gene = clicked_row().get("gene_name", "images") if clicked_row() else "images"
        try:
            choice = input.dl_image_choice()
        except Exception:
            choice = "__all__"
        if choice == "__all_tiff__":
            return f"{gene}_images.zip"
        if choice == "__collage__":
            return f"{gene}_collage.png"
        if choice.startswith("tiff_"):
            idx = int(choice[5:])
            if idx < len(images):
                safe = re.sub(r"[^\w\-.]", "_", images[idx][0])
                return f"{safe}.tiff"
        return f"{gene}_image.tiff"

    @render.download(filename=_dl_filename)
    def dl_images():
        import time as _time
        flagged_now = flagged_mask_paths()
        images = [t for t in _displayed_images() if str(t[2]) not in flagged_now]
        if not images:
            yield b""
            return
        try:
            choice = input.dl_image_choice()
        except Exception:
            choice = "__all_tiff__"

        if choice == "__all_tiff__":
            exp_dir = input.exp_dir().strip()
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for lbl, _data_uri, mask_path_str in images:
                    raw_path = mask_path_to_raw(mask_path_str, exp_dir)
                    safe_name = re.sub(r"[^\w\-.]", "_", lbl) + ".tiff"
                    if raw_path and raw_path.exists():
                        zf.write(str(raw_path), safe_name)
            yield buf.getvalue()

        elif choice == "__collage__":
            from PIL import Image as PILImage, ImageDraw, ImageFont
            from collections import OrderedDict

            pil_imgs = []
            parsed_labels = []
            for lbl, data_uri, _mpath in images:
                raw_bytes = base64.b64decode(data_uri.split(",", 1)[1])
                pil_imgs.append(PILImage.open(io.BytesIO(raw_bytes)).convert("RGB"))
                parts = [p.strip() for p in lbl.split("|")]
                meta = {"gene": parts[0] if parts else "",
                        "strain": "", "lib": "", "well": "", "day": ""}
                for p in parts[1:]:
                    if p.startswith("Strain "):
                        meta["strain"] = p[7:]
                    elif p.startswith("Well "):
                        meta["well"] = p[5:]
                    elif p.startswith("Day "):
                        meta["day"] = p[4:]
                    else:
                        meta["lib"] = p
                parsed_labels.append(meta)

            if not pil_imgs:
                yield b""
                return

            groups: dict[str, dict[str, list]] = OrderedDict()
            for im, meta in zip(pil_imgs, parsed_labels):
                s  = meta.get("strain") or "—"
                lb = meta.get("lib")    or "—"
                groups.setdefault(s, OrderedDict()).setdefault(lb, []).append(
                    (im, meta.get("well", ""))
                )

            w, h = pil_imgs[0].size
            COLS = 4
            pad  = max(12, w // 55)

            _bold = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
            _reg  = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]

            def _try_font(paths, size):
                for fp in paths:
                    try:
                        return ImageFont.truetype(fp, size)
                    except Exception:
                        pass
                return ImageFont.load_default()

            sz_title  = max(36, w // 13)
            sz_strain = max(30, w // 17)
            sz_lib    = max(24, w // 22)
            sz_cap    = max(22, w // 25)

            f_title  = _try_font(_bold, sz_title)
            f_strain = _try_font(_bold, sz_strain)
            f_lib    = _try_font(_reg,  sz_lib)
            f_cap    = _try_font(_reg,  sz_cap)

            title_h      = sz_title  + pad * 4
            strain_bar_h = sz_strain + pad * 2
            lib_h        = sz_lib    + pad
            cap_h        = sz_cap    + pad      
            img_row_h    = h + cap_h + pad

            total_w = pad + COLS * (w + pad)
            total_h = title_h + pad

            for _s, libs in groups.items():
                total_h += strain_bar_h
                for _lb, img_list in libs.items():
                    n_rows = (len(img_list) + COLS - 1) // COLS
                    total_h += lib_h + n_rows * img_row_h

            BG           = (245, 245, 245)
            C_TITLE      = (15,  15,  15)
            C_STRAIN_BG  = (50,  80, 130)
            C_STRAIN_FG  = (255, 255, 255)
            C_LIB        = (70,  70,  70)
            C_CAP        = (55,  55,  55)
            C_SEP        = (190, 190, 190)

            grid = PILImage.new("RGB", (total_w, total_h), color=BG)
            draw = ImageDraw.Draw(grid)

            gene_txt  = (parsed_labels[0].get("gene") or "") if parsed_labels else ""
            day_txt   = (parsed_labels[0].get("day")  or "") if parsed_labels else ""
            title_txt = f"{gene_txt}   |   Day {day_txt}" if day_txt else gene_txt
            draw.text((pad, pad), title_txt, fill=C_TITLE, font=f_title)
            y_cur = title_h
            draw.line([(0, y_cur), (total_w, y_cur)], fill=C_SEP, width=3)
            y_cur += pad

            for s, libs in groups.items():
                draw.rectangle([(0, y_cur), (total_w, y_cur + strain_bar_h)],
                                fill=C_STRAIN_BG)
                draw.text((pad, y_cur + pad // 2),
                          f"Strain {s}", fill=C_STRAIN_FG, font=f_strain)
                y_cur += strain_bar_h

                for lb, img_list in libs.items():
                    draw.text((pad * 2, y_cur + pad // 4), lb, fill=C_LIB, font=f_lib)
                    y_cur += lib_h

                    n_rows = (len(img_list) + COLS - 1) // COLS
                    for k, (im, well) in enumerate(img_list):
                        r_i, c_i = divmod(k, COLS)
                        x = pad + c_i * (w + pad)
                        y = y_cur + r_i * img_row_h
                        grid.paste(im, (x, y))
                        if well:
                            draw.text((x + pad // 2, y + h + pad // 4),
                                      f"Well {well}", fill=C_CAP, font=f_cap)
                    y_cur += n_rows * img_row_h

            buf = io.BytesIO()
            grid.save(buf, format="PNG")
            yield buf.getvalue()

        elif choice.startswith("tiff_"):
            idx = int(choice[5:])
            exp_dir = input.exp_dir().strip()
            if idx < len(images):
                _lbl, _data_uri, mask_path_str = images[idx]
                raw_path = mask_path_to_raw(mask_path_str, exp_dir)
                if raw_path and raw_path.exists():
                    yield raw_path.read_bytes()
                    return
            yield b""

        else:
            yield b""

    @reactive.effect
    @reactive.event(input.close_image_btn)
    def _close():
        clicked_row.set(None)

    @render.ui
    def selected_info():
        row = clicked_row()
        if row is None:
            return ui.p("Click a point on the plot to view its image.",
                        style="color:#888;")

        gene = row.get("gene_name", "?")
        well = row.get("well96", "?")
        plate = row.get("plate", "?")

        exp_dir = input.exp_dir().strip()
        img_ui = ui.p(
            "(Set experiment directory to view images.)",
            style="color:#aaa; font-size:0.85em;",
        )

        if exp_dir:
            df = raw_df()
            ch = int(input.image_channel()) if hasattr(input, "image_channel") else 1
            fcol = display_file_col(df, ch) if df is not None else None
            if df is not None and fcol is not None:
                match_filter = gene_name_match_expr(gene)
                if plate not in (None, "", "?"):
                    match_filter = match_filter & (
                        pl.col("plate").cast(pl.Utf8) == str(plate)
                    )

                plot_type = input.plot_type()
                if plot_type in ("strain_linear", "strain_log", "log2fc_scatter"):
                    try:
                        cmp_day = str(input.cmp_day())
                    except Exception:
                        cmp_day = ""
                    active_days = [cmp_day] if cmp_day and df_has_meaningful_day(df) else []
                else:
                    active_days = (
                        [str(d) for d in meaningful_screen_days(df)]
                        if df_has_meaningful_day(df) and hasattr(input, "day_filter")
                        else []
                    )
                    try:
                        selected = list(input.day_filter())
                        if selected:
                            active_days = [str(d) for d in selected]
                    except Exception:
                        pass

                if active_days and df_has_meaningful_day(df) and "day" in df.columns:
                    match_filter = match_filter & pl.col("day").cast(pl.Utf8).is_in(active_days)

                match = df.filter(match_filter)
                try:
                    global_lib_sel = input.analysis_library()
                except Exception:
                    global_lib_sel = "__all__"
                # Plot uses synthetic __control__ lib_code; CSV rows keep real lib codes.
                if is_plot_synthetic_control_row(row):
                    match = apply_library_filter(match, global_lib_sel)
                elif global_lib_sel == "__all__":
                    clicked_lib_group = row.get("library_group", None)
                    if clicked_lib_group in (None, "", "__all__"):
                        clicked_lib_group = library_group_from_code(
                            row.get("lib_code", None)
                        )
                    if clicked_lib_group not in (None, "", "__all__"):
                        match = apply_library_filter(match, str(clicked_lib_group))
                    else:
                        match = apply_library_filter(match, global_lib_sel)
                else:
                    match = apply_library_filter(match, global_lib_sel)

                if is_plot_synthetic_control_row(row):
                    if "is_control" in match.columns:
                        match = match.filter(
                            pl.col("is_control").cast(pl.Utf8).str.to_lowercase() == "true"
                        )
                elif (
                    str(row.get("lib_code") or "").strip()
                    not in ("", "__control__")
                    and "lib_code" in match.columns
                ):
                    match = match.filter(
                        pl.col("lib_code").cast(pl.Utf8)
                        == str(row.get("lib_code")).strip()
                    )

                _displayed_imgs: list[tuple[str, str, str]] = []
                combined_flag_map: dict[str, str] = {}
                if match.height > 0:
                    import time as _time
                    ch = int(input.image_channel()) if hasattr(input, "image_channel") else 1
                    is_comparison = plot_type in ("strain_linear", "strain_log", "log2fc_scatter")

                    try:
                        lib_filter = input.lib_filter()
                    except Exception:
                        lib_filter = "__all__"
                    all_libs = sorted(
                        match["lib_code"].cast(pl.Utf8).unique().drop_nulls().to_list()
                    ) if "lib_code" in match.columns else []
                    if (
                        global_lib_sel == "__all__"
                        and lib_filter != "__all__"
                        and lib_filter in all_libs
                        and "lib_code" in match.columns
                    ):
                        match = match.filter(pl.col("lib_code").cast(pl.Utf8) == lib_filter)

                    cap_group = (
                        ["strain"]
                        if is_comparison and "strain" in match.columns
                        else None
                    )
                    if is_plot_synthetic_control_row(row) or match.height > (
                        MAX_PRELOAD_PER_PLOTTED_POINT * (2 if is_comparison else 1)
                    ):
                        match = cap_image_rows(
                            match,
                            fcol,
                            max_rows=MAX_PRELOAD_PER_PLOTTED_POINT,
                            group_cols=cap_group,
                        )

                    def render_strain_images(strain_df, strain_label):
                        """Render images for one strain grouped by library."""
                        imgs_out, missing_out = [], []
                        local_flag_map: dict[str, str] = {}
                        libs = sorted(
                            strain_df["lib_code"].cast(pl.Utf8).unique().drop_nulls().to_list()
                        ) if "lib_code" in strain_df.columns else [None]

                        for lib_code in libs:
                            if lib_code is not None and "lib_code" in strain_df.columns:
                                lib_df = strain_df.filter(
                                    pl.col("lib_code").cast(pl.Utf8) == lib_code
                                )
                                lib_header = ui.p(
                                    f"Library: {lib_code}",
                                    style="font-size:0.78em; color:#888; font-style:italic; margin:3px 0 1px 0;",
                                )
                            else:
                                lib_df = strain_df
                                lib_header = None

                            rows = lib_df.unique(subset=[fcol]) if fcol in lib_df.columns else lib_df
                            lib_imgs_flagged = []
                            lib_imgs_clean = []
                            t0 = _time.perf_counter()
                            for r in rows.iter_rows(named=True):
                                mask_file = r[fcol]
                                raw_path = mask_path_to_raw(mask_file, exp_dir)
                                well_lbl = well384_label(r.get('row384', ''), r.get('col384', '')) \
                                    if r.get('row384', '') not in ('', None) else r.get('well96', '')
                                lib = r.get('lib_code', '')
                                lib_tag = f" | {lib}" if lib else ""
                                lbl = f"{strain_label}{lib_tag} | Well {well_lbl} | Day {r.get('day', '')}"
                                overlay_mode = str(input.overlay_mode()) if hasattr(input, "overlay_mode") else "none"
                                alpha = float(input.overlay_alpha()) if hasattr(input, "overlay_alpha") else 0.35
                                obj_col = f"{fcol[:-5]}_object_csv" if fcol.endswith("_file") else None
                                obj_csv_raw = r.get(obj_col) if obj_col else None
                                mask_mode = "eq1"  # match pipeline QC object_label
                                cache_key = _cache_key(
                                    mask_file,
                                    ch,
                                    overlay_mode,
                                    alpha,
                                    obj_csv_raw,
                                    mask_mode,
                                )
                                data_uri = _image_cache.get(cache_key)
                                if data_uri is None and raw_path and raw_path.exists():
                                    t_read = _time.perf_counter()
                                    if overlay_mode != "none":
                                        mask_p = _resolve_existing_path(mask_file, exp_dir)
                                        obj_csv_p = _resolve_existing_path(obj_csv_raw, exp_dir) if obj_col else None
                                        if mask_p and mask_p.exists():
                                            data_uri = overlay_mask_on_raw_to_base64_jpeg(
                                                raw_path,
                                                mask_p,
                                                channel=ch,
                                                alpha=alpha,
                                                mode=overlay_mode,
                                                object_csv_path=obj_csv_p,
                                                mask_positive_mode=mask_mode,
                                            )
                                        else:
                                            data_uri = tiff_to_base64_jpeg(raw_path, channel=ch)
                                    else:
                                        data_uri = tiff_to_base64_jpeg(raw_path, channel=ch)
                                    print(f"[profile] preview_build: {_time.perf_counter()-t_read:.2f}s — {raw_path.name}", flush=True)
                                    if data_uri:
                                        with _cache_lock:
                                            _image_cache[cache_key] = data_uri
                                if data_uri:
                                    _displayed_imgs.append((lbl, data_uri, mask_file))
                                    mask_str = str(mask_file)
                                    flag_id = "flag_" + hashlib.md5(mask_str.encode("utf-8")).hexdigest()[:12]
                                    local_flag_map[flag_id] = mask_str
                                    is_flagged = mask_str in flagged_mask_paths()
                                    image_block = ui.div(
                                        ui.input_checkbox(
                                            flag_id,
                                            "Flag this image",
                                            value=is_flagged,
                                        ),
                                        ui.p(lbl, style="font-size:0.75em; color:#555; margin:2px 0;"),
                                        ui.div(
                                            ui.tags.img(
                                                src=data_uri,
                                                style="max-width:100%; border:1px solid #ddd; border-radius:4px; margin-bottom:6px;",
                                            ),
                                            ui.div(
                                                "FLAGGED",
                                                style=(
                                                    "position:absolute; top:8px; left:8px; background:rgba(180,0,0,0.9); "
                                                    "color:#fff; font-size:0.68em; font-weight:700; padding:2px 6px; "
                                                    "border-radius:3px; letter-spacing:0.3px;"
                                                ),
                                            ) if is_flagged else ui.div(),
                                            ui.div(
                                                style=(
                                                    "position:absolute; inset:0; background:rgba(255,0,0,0.18); "
                                                    "border:2px solid rgba(180,0,0,0.75); border-radius:4px; pointer-events:none;"
                                                ),
                                            ) if is_flagged else ui.div(),
                                            style="position:relative; display:inline-block; width:100%;",
                                        ),
                                        style=(
                                            "padding:4px 6px 8px 6px; border-radius:6px; margin-bottom:6px;"
                                            + (
                                                " border:1px solid #cc4444; background:#fff5f5;"
                                                if is_flagged
                                                else " border:1px solid #e8e8e8;"
                                            )
                                        ),
                                    )
                                    if is_flagged:
                                        lib_imgs_flagged.append(image_block)
                                    else:
                                        lib_imgs_clean.append(image_block)
                                else:
                                    missing_out.append(
                                        f"{lbl} — {'decode error' if (raw_path and raw_path.exists()) else f'file not found: {raw_path}'}"
                                    )
                            print(f"[profile] render_strain_images lib={lib_code}: {_time.perf_counter()-t0:.2f}s total", flush=True)
                            lib_imgs = []
                            if lib_imgs_flagged:
                                lib_imgs.extend([
                                    ui.p(
                                        "Flagged wells",
                                        style="font-size:0.76em; color:#b00000; font-weight:700; margin:4px 0 2px 0;",
                                    ),
                                    *lib_imgs_flagged,
                                ])
                            if lib_imgs_clean:
                                lib_imgs.extend([
                                    ui.p(
                                        "Non-flagged wells",
                                        style="font-size:0.76em; color:#666; font-weight:700; margin:4px 0 2px 0;",
                                    ),
                                    *lib_imgs_clean,
                                ])
                            if lib_imgs:
                                block = [lib_header] + lib_imgs if lib_header else lib_imgs
                                imgs_out.extend(block)
                        return imgs_out, missing_out, local_flag_map

                    imgs = []
                    not_found = []

                    if "strain" in match.columns:
                        if is_comparison:
                            strains_to_show = [str(input.strain_a()), str(input.strain_b())]
                        elif plot_type == "volcano":
                            try:
                                v_mode = input.volcano_mode()
                                v_strain = str(input.volcano_strain())
                            except Exception:
                                v_mode, v_strain = "combined", None
                            if v_mode == "per_strain" and v_strain:
                                strains_to_show = [v_strain]
                            else:
                                strains_to_show = sorted(
                                    match["strain"].cast(pl.Utf8).unique().to_list()
                                )
                        else:
                            strains_to_show = sorted(
                                match["strain"].cast(pl.Utf8).unique().to_list()
                            )
                        for idx_s, s in enumerate(strains_to_show):
                            strain_match = match.filter(pl.col("strain").cast(pl.Utf8) == s)
                            s_imgs, s_missing, s_flag_map = render_strain_images(
                                strain_match, f"{gene} | Strain {s}"
                            )
                            if s_imgs:
                                divider = ui.div(
                                    ui.p(f"── Strain {s} ──",
                                         style="font-weight:bold; font-size:0.85em; color:#fff; margin:0; padding:3px 8px;"),
                                    style="background:#457b9d; border-radius:4px; margin:8px 0 4px 0;",
                                )
                                imgs.append(ui.div(divider, *s_imgs))
                            not_found.extend(s_missing)
                            combined_flag_map.update(s_flag_map)
                    else:
                        shown = match.unique(subset=[fcol]) if fcol in match.columns else match
                        s_imgs, s_missing, s_flag_map = render_strain_images(shown, gene)
                        imgs.extend(s_imgs)
                        not_found.extend(s_missing)
                        combined_flag_map.update(s_flag_map)

                    lib_selector = ui.div()
                    if global_lib_sel == "__all__" and all_libs and len(all_libs) > 1:
                        lib_choices = {"__all__": "Both libraries"}
                        for lb in all_libs:
                            lib_choices[lb] = lb
                        lib_selector = ui.div(
                            ui.input_radio_buttons(
                                "lib_filter", "Library",
                                choices=lib_choices,
                                selected=lib_filter,
                                inline=True,
                            ),
                            style="margin-bottom:6px;",
                        )

                    n_images = len(_displayed_imgs) + len(not_found)
                    flagged_now = flagged_mask_paths()
                    flagging_ui = ui.div(
                        ui.p(
                            f"Flag problematic images: {len(flagged_now)} total flagged globally.",
                            style="font-size:0.8em; color:#666; margin:4px 0 4px 0;",
                        ),
                        ui.div(
                            ui.input_action_button(
                                "clear_all_flags_btn",
                                "Clear All Flags",
                                class_="btn btn-sm btn-outline-secondary",
                            ),
                            style="display:flex; gap:6px; margin:6px 0 8px 0; flex-wrap:wrap;",
                        ),
                    )
                    summary = ui.p(
                        f"Showing {len(_displayed_imgs)} of {n_images} images" +
                        (f"  |  {len(not_found)} missing" if not_found else ""),
                        style="font-size:0.8em; color:#888; margin:2px 0 6px 0;",
                    )
                    missing_details = ui.div(
                        *[ui.p(m, style="font-size:0.75em; color:#c00; margin:1px 0;")
                          for m in not_found]
                    )
                    img_ui = ui.div(lib_selector, flagging_ui, summary, missing_details, *imgs) if imgs else ui.div(
                        lib_selector, flagging_ui, summary, missing_details,
                        ui.p("No images could be displayed.", style="color:#c00; font-size:0.85em;"),
                    )

                    _displayed_images.set(_displayed_imgs)
                    flag_input_map.set(combined_flag_map)
                else:
                    flag_input_map.set({})
                    img_ui = ui.p(
                        "No matching images in the loaded CSV for this point "
                        "(check experiment directory, day filter, and library selection).",
                        style="color:#c00; font-size:0.85em;",
                    )

        return img_ui
