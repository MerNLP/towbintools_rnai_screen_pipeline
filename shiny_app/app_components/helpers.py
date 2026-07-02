"""Shared helpers for the RNAi screen Shiny dashboard."""

import os
import re
import sys
from pathlib import Path

import polars as pl

_PIPELINE_SCRIPTS = Path(__file__).resolve().parents[2] / "pipeline_scripts"
if str(_PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_SCRIPTS))
from screen_utils import find_exp_plate_subfolder_in_parts  # noqa: E402

_COMBINED_CSV_SKIP_DIRS = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    "raw",
    "temp_files",
    "crops_image",
})

# Skip seg/object trees when scanning for combined CSVs.
_COMBINED_CSV_SKIP_DIR_RE = re.compile(
    r"^(ch\d+_seg(?:_str)?|.*_object_csvs)$",
    re.IGNORECASE,
)

META_COLS = {
    "strain",
    "day",
    "plate",
    "row384",
    "col384",
    "well96",
    "lib_code",
    "gene_name",
    "is_control",
}
META_SUFFIXES = ("_file", "_excluded_small")

GENE_LEVEL_KEY_COLS = [
    "strain",
    "day",
    "plate",
    "gene_name",
    "lib_code",
    "is_control",
]

_MAX_DEVELOPMENTAL_DAY = 9999  # filter out YYYYMMDD values mis-labeled as day


def meaningful_screen_days(df: pl.DataFrame | None) -> list:
    """Return developmental day values worth filtering on."""
    if df is None or "day" not in df.columns:
        return []
    out: list = []
    for d in df["day"].drop_nulls().unique().to_list():
        try:
            v = float(d)
        except (TypeError, ValueError):
            continue
        if 0 < v <= _MAX_DEVELOPMENTAL_DAY:
            out.append(d)
    return sorted(out, key=lambda x: float(x))


def df_has_meaningful_day(df: pl.DataFrame | None) -> bool:
    return len(meaningful_screen_days(df)) > 0


def screen_group_keys(df: pl.DataFrame, *cols: str) -> list[str]:
    """Filter grouping columns to those present; drop day when not meaningful."""
    keys = [c for c in cols if c in df.columns]
    if "day" in keys and not df_has_meaningful_day(df):
        keys.remove("day")
    return keys


def gene_level_key_cols(df: pl.DataFrame) -> list[str]:
    """Group-by keys for gene-level tables, omitting day when not meaningful."""
    keys = [c for c in GENE_LEVEL_KEY_COLS if c in df.columns]
    if "day" in keys and not df_has_meaningful_day(df):
        keys.remove("day")
    return keys


def numeric_cols(df: pl.DataFrame) -> list[str]:
    """Numeric columns suitable for metrics / aggregation (excludes metadata and join dups)."""
    return [
        c
        for c in df.columns
        if c not in META_COLS
        and not any(c.endswith(s) for s in META_SUFFIXES)
        and not c.endswith("_right")
        and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
    ]


def file_col(df: pl.DataFrame) -> str | None:
    for c in df.columns:
        if c.endswith("_file"):
            return c
    return None


def display_file_col(df: pl.DataFrame, image_channel: int | None) -> str | None:
    """Choose the segmentation file column that matches the displayed raw channel."""
    if df is None:
        return None
    ch = int(image_channel) if image_channel is not None else 1
    preferred = "ch1_seg_file" if ch == 0 else "ch2_seg_file"
    if preferred in df.columns:
        return preferred
    return file_col(df)


def guess_col(df: pl.DataFrame, *keywords: str) -> str | None:
    """Return the first column whose name contains ALL keywords (case-insensitive)."""
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None


def guess_col_among(columns: list[str], *keywords: str) -> str | None:
    """Like guess_col but restricted to an ordered list of column names (e.g. metric choices)."""
    for c in columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None


def library_group_expr(col: str = "lib_code") -> pl.Expr:
    """Map raw lib_code values to high-level library groups."""
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    s_lower = s.str.to_lowercase()
    return (
        pl.when(s_lower.is_in(sorted(_CONTROL_LIB_ALIASES_LOWER)))
        .then(pl.lit("control"))
        .when(s.str.starts_with("GHR-"))
        .then(pl.lit("ghr"))
        .when(s.str.contains(r"^\d+@"))
        .then(pl.lit("numeric"))
        .otherwise(pl.lit("other"))
    )


def apply_library_filter(df: pl.DataFrame, selected: str | None) -> pl.DataFrame:
    """Filter by high-level library group; include controls for each library view."""
    if df is None or "lib_code" not in df.columns:
        return df
    sel = str(selected or "__all__")
    if sel in ("__all__", ""):
        return df
    tmp = "__lib_group"
    df2 = df.with_columns(library_group_expr("lib_code").alias(tmp))
    if sel in ("numeric", "ghr"):
        df2 = df2.filter(pl.col(tmp).is_in([sel, "control"]))
    else:
        df2 = df2.filter(pl.col(tmp) == sel)
    return df2.drop(tmp)


CONTROL_LIB_BUCKET = "__control__"

_EMPTY_VECTOR_ALIASES = frozenset({"empty", "empty_vector", "l4440"})
_CONTROL_LIB_ALIASES_LOWER = frozenset(
    {"ama-1", "ama_1", "mex-3", "mex_3", "empty", "empty_vector", "l4440", "ev"}
)


def canonical_control_gene_expr(col: str = "gene_name") -> pl.Expr:
    """Map control gene naming variants to EV / empty / ama-1 / mex-3."""
    gn = pl.col(col).cast(pl.Utf8).str.strip_chars()
    gn_l = gn.str.to_lowercase()
    return (
        pl.when(gn_l == "ev")
        .then(pl.lit("EV"))
        .when(gn_l.is_in(sorted(_EMPTY_VECTOR_ALIASES)))
        .then(pl.lit("empty"))
        .when(gn_l.is_in(["mex-3", "mex_3"]))
        .then(pl.lit("mex-3"))
        .when(gn_l.is_in(["ama-1", "ama_1"]))
        .then(pl.lit("ama-1"))
        .otherwise(gn)
    )


def is_ev_gene_expr(col: str = "gene_name") -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase() == "ev"


def is_empty_vector_gene_expr(col: str = "gene_name") -> pl.Expr:
    gn_l = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return gn_l.is_in(sorted(_EMPTY_VECTOR_ALIASES))


def is_control_flag_expr(col: str = "is_control") -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.to_lowercase() == "true"


def select_normalization_reference_df(df: pl.DataFrame) -> pl.DataFrame:
    """Rows for fold-change baseline: EV, else empty-vector controls, else all controls."""
    if df is None or df.height == 0:
        return df
    ev = df.filter(is_ev_gene_expr())
    if ev.height > 0:
        return ev
    empty_vec = df.filter(is_empty_vector_gene_expr())
    if empty_vec.height > 0:
        return empty_vec
    if "is_control" in df.columns:
        return df.filter(is_control_flag_expr())
    return df


_GENE_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "ev": ("ev",),
    "empty": ("empty", "l4440", "empty_vector"),
    "empty_vector": ("empty", "l4440", "empty_vector"),
    "l4440": ("empty", "l4440", "empty_vector"),
    "mex-3": ("mex-3", "mex_3"),
    "mex_3": ("mex-3", "mex_3"),
    "ama-1": ("ama-1", "ama_1"),
    "ama_1": ("ama-1", "ama_1"),
}


def gene_name_match_expr(gene: str) -> pl.Expr:
    """Match a plotted control/gene name to CSV gene_name values (with aliases)."""
    key = str(gene).strip().lower()
    aliases = _GENE_NAME_ALIASES.get(key, (key,))
    return (
        pl.col("gene_name")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .is_in([a.lower() for a in aliases])
    )


def is_plot_synthetic_control_row(row: dict) -> bool:
    """True for strain-comparison control points (synthetic lib bucket)."""
    if str(row.get("lib_code") or "").strip() == CONTROL_LIB_BUCKET:
        return True
    return False


PRELOAD_JOIN_KEY_CANDIDATES = (
    "gene_name",
    "strain",
    "day",
    "plate",
    "lib_code",
    "is_control",
    "well96",
    "row384",
    "col384",
)


def preload_join_key_cols(plot_df: pl.DataFrame, source_df: pl.DataFrame) -> list[str]:
    """Join keys for matching plot points to per-image CSV rows."""
    key_cols = [
        c
        for c in PRELOAD_JOIN_KEY_CANDIDATES
        if c in plot_df.columns and c in source_df.columns
    ]
    # Synthetic control bucket on plot rows does not match CSV lib_code.
    if "lib_code" in key_cols and "lib_code" in plot_df.columns:
        has_bucket = (
            plot_df.filter(pl.col("lib_code").cast(pl.Utf8) == CONTROL_LIB_BUCKET).height
            > 0
        )
        if has_bucket:
            key_cols = [c for c in key_cols if c != "lib_code"]
    return key_cols


def cap_image_rows(
    df: pl.DataFrame,
    file_col: str,
    *,
    max_rows: int,
    group_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Deterministically limit image rows (e.g. control wells) for the viewer."""
    if df is None or df.height == 0 or file_col not in df.columns:
        return df
    order_cols = [
        c
        for c in [
            "strain",
            "plate",
            "well96",
            "row384",
            "col384",
            "lib_code",
            file_col,
        ]
        if c in df.columns
    ]
    sorted_df = df.sort(order_cols) if order_cols else df
    use_groups = [c for c in (group_cols or []) if c in sorted_df.columns]
    if use_groups:
        return (
            sorted_df.with_columns(
                pl.col(file_col).cum_count().over(use_groups).alias("__img_idx")
            )
            .filter(pl.col("__img_idx") <= max_rows)
            .drop("__img_idx")
        )
    return sorted_df.unique(subset=[file_col]).head(max_rows)


def library_group_from_code(code: str | None) -> str:
    """Infer high-level library group from a single lib_code value."""
    if code is None:
        return "other"
    s = str(code).strip()
    s_lower = s.lower()
    if s_lower in _CONTROL_LIB_ALIASES_LOWER or s_lower == CONTROL_LIB_BUCKET:
        return "control"
    if s.startswith("GHR-"):
        return "ghr"
    if re.match(r"^\d+@", s):
        return "numeric"
    return "other"


def _resolve_existing_path(path_str: str | None, experiment_dir: str | None = None) -> Path | None:
    if not path_str:
        return None
    p = Path(str(path_str))
    if p.exists():
        return p
    if experiment_dir:
        cand = Path(experiment_dir) / str(path_str)
        if cand.exists():
            return cand
    return None


def selected_metric_col(df: pl.DataFrame, metric_col: str | None) -> str | None:
    """Return selected metric column name if available in the dataframe."""
    if df is None or not metric_col:
        return None
    if metric_col not in df.columns:
        return None
    return metric_col


def is_gene_level_metric_col(col: str) -> bool:
    """True for pipeline gene-level columns (e.g. avg_*_um2_gene or *_qc_gene)."""
    name = str(col).lower()
    return name.endswith("_gene") or "_qc_gene" in name


def metric_col_choices(df: pl.DataFrame) -> list[str]:
    """Numeric metric columns, with gene-level columns listed first."""
    cols = numeric_cols(df)
    gene_cols = [c for c in cols if is_gene_level_metric_col(c)]
    other = [c for c in cols if c not in gene_cols]
    return gene_cols + other


def default_metric_col(df: pl.DataFrame) -> str | None:
    """Pick a notebook-style gene-level metric when available."""
    cols = metric_col_choices(df)
    if not cols:
        return None
    for keywords in (
        ("avg", "um2", "qc", "gene"),
        ("avg", "um2", "gene"),
        ("avg", "per", "um2"),
        ("per", "um2"),
    ):
        hit = guess_col_among(cols, *keywords)
        if hit:
            return hit
    return cols[0]


def prepare_comparison_metrics_df(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize control aliases and collapse controls to one lib bucket for plotting."""
    if df is None:
        return df
    df = df.with_columns(canonical_control_gene_expr("gene_name").alias("gene_name"))
    if "lib_code" in df.columns and "is_control" in df.columns:
        is_ctrl = is_control_flag_expr()
        df = df.with_columns(
            pl.when(is_ctrl)
            .then(pl.lit(CONTROL_LIB_BUCKET))
            .otherwise(pl.col("lib_code").cast(pl.Utf8))
            .alias("lib_code")
        )
    return df


def collapse_to_gene_level(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    """One row per gene key; gene-level columns are first(), others are mean()."""
    keys = gene_level_key_cols(df)
    if not keys or metric_col not in df.columns:
        return df

    if is_gene_level_metric_col(metric_col):
        return df.group_by(keys, maintain_order=True).agg(
            pl.col(metric_col).first().alias(metric_col)
        )

    return df.group_by(keys, maintain_order=True).agg(
        pl.col(metric_col).mean().alias(metric_col)
    )


def infer_experiment_dir_from_df(df: pl.DataFrame) -> str | None:
    """Guess experiment root from mask paths in a combined CSV."""
    fcol = file_col(df)
    if df is None or not fcol or fcol not in df.columns or df.height == 0:
        return None
    sample = df.select(pl.col(fcol).drop_nulls().head(1)).item()
    if not sample:
        return None
    parts = Path(str(sample)).parts
    subfolder = find_exp_plate_subfolder_in_parts(parts)
    if not subfolder:
        return None
    m = re.match(r"^(\d{8})_wBT", str(subfolder))
    if not m:
        return None
    date = m.group(1)
    agraf_root = Path("/mnt/towbin.data/shared/agraf")
    if not agraf_root.is_dir():
        return None
    for cand in sorted(agraf_root.glob(f"{date}_Kinetix*")):
        if (cand / "raw" / subfolder).is_dir():
            return str(cand)
    return None


def list_dir(path: Path):
    """Return (dirs, csv_files) in path."""
    try:
        entries = sorted(path.iterdir())
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
        csvs = [e for e in entries if e.is_file() and e.suffix.lower() in (".csv", ".parquet")]
        return dirs, csvs
    except Exception:
        return [], []


def well384_label(row384, col384) -> str:
    """Format a 384-well label from row+col."""
    try:
        return f"{str(row384).strip()}{int(float(col384)):02d}"
    except Exception:
        return f"{row384}{col384}"


def _is_combined_table_name(name: str) -> bool:
    return (
        name.endswith("_combined.csv")
        or name.endswith("_combined.parquet")
        or name.endswith("_combined")
    )


def _should_skip_combined_scan_dir(dirname: str) -> bool:
    if dirname in _COMBINED_CSV_SKIP_DIRS or dirname.startswith("."):
        return True
    return _COMBINED_CSV_SKIP_DIR_RE.match(dirname) is not None


def find_combined_csvs(exp_dir: str) -> list[Path]:
    """Find *_combined tables under an experiment directory (pruned walk)."""
    try:
        root = Path(exp_dir).expanduser()
        if not root.is_dir():
            return []

        candidates: list[Path] = []
        seen: set[Path] = set()

        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = [
                d for d in dirnames if not _should_skip_combined_scan_dir(d)
            ]
            cur = Path(dirpath)
            for fn in filenames:
                if _is_combined_table_name(fn):
                    path = (cur / fn).resolve()
                    if path not in seen:
                        seen.add(path)
                        candidates.append(path)

        return sorted(candidates)
    except Exception:
        return []
