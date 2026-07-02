"""Body/pharynx mask measurements and well/gene aggregation for RNAi screens."""

import os
import re

import numpy as np
import polars as pl
import tifffile as tiff
from joblib import Parallel, delayed, parallel_config
from scipy import ndimage as ndi


def extract_channel_tag(root_path):
    """Extract channel tag from a mask root path (e.g. ch2_seg)."""
    parts = str(root_path).split("/")
    if len(parts) > 1 and "analysis" in parts[0]:
        parts = parts[1:]
    return os.path.basename(os.path.normpath(os.path.join(*parts)))


def extract_primary_channel_tag(root_path):
    """Return the base channel tag, e.g. ch1_seg or ch2_seg."""
    tag = extract_channel_tag(root_path)
    match = re.search(r"ch\d+_seg", tag)
    return match.group(0) if match else tag


DEFAULT_MIN_BODY_AREA_PX = 100
DEFAULT_MAX_BODY_AREA_PX = 1_000_000
DEFAULT_MIN_PHARYNX_AREA_PX = 100
DEFAULT_MAX_PHARYNX_AREA_PX = 10_000

GENE_LEVEL_KEY_COLS = [
    "strain",
    "day",
    "plate",
    "gene_name",
    "lib_code",
    "is_control",
]

WELL_POSITION_KEY_COLS = [
    "strain",
    "day",
    "plate",
    "row384",
    "col384",
    "well96",
]

GENE_ANNOTATION_COLS = ["lib_code", "gene_name", "is_control"]


def _well_position_key_cols(*dfs):
    """Well-position keys present in all non-null frames."""
    present = set(WELL_POSITION_KEY_COLS)
    for df in dfs:
        if df is not None:
            present &= set(df.columns)
    return [c for c in WELL_POSITION_KEY_COLS if c in present]


def _join_key_cols(*dfs, base_cols):
    """Join/sort keys present in all provided frames."""
    present = set(base_cols)
    for df in dfs:
        if df is not None:
            present &= set(df.columns)
    return [c for c in base_cols if c in present]


def _sort_keys(df, *cols):
    return [c for c in cols if c in df.columns]


def _screen_row_meta(row):
    """Metadata copied from a screen filemap row."""
    meta = {
        "strain": row["strain"],
        "plate": row["plate"],
        "row384": row.get("row384", None),
        "col384": row.get("col384", None),
        "well96": row.get("well96", None),
        "lib_code": row.get("lib_code", None),
        "gene_name": row.get("gene_name", None),
        "is_control": row.get("is_control", None),
    }
    if row.get("day") is not None:
        meta["day"] = row["day"]
    return meta


def _measure_connected_components(binary_mask, min_area_px, max_area_px=float("inf")):
    """Return one record per connected component in a binary mask."""
    structure = np.ones((3, 3), dtype=bool)
    lbl, n = ndi.label(binary_mask, structure=structure)
    if n == 0:
        return []

    object_slices = ndi.find_objects(lbl)
    records = []
    for object_idx, obj_slice in enumerate(object_slices, start=1):
        if obj_slice is None:
            continue
        region = lbl[obj_slice] == object_idx
        area_px = float(region.sum())
        ys, xs = np.nonzero(region)
        min_row, max_row = obj_slice[0].start, obj_slice[0].stop
        min_col, max_col = obj_slice[1].start, obj_slice[1].stop
        centroid_y = float(min_row + ys.mean()) if ys.size else np.nan
        centroid_x = float(min_col + xs.mean()) if xs.size else np.nan
        records.append(
            {
                "object_label": int(object_idx),
                "area_px": area_px,
                "bbox_min_row": int(min_row),
                "bbox_min_col": int(min_col),
                "bbox_max_row": int(max_row),
                "bbox_max_col": int(max_col),
                "bbox_area_px": float((max_row - min_row) * (max_col - min_col)),
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "kept_after_area_filter": bool(
                    min_area_px <= area_px <= max_area_px
                ),
            }
        )
    return records


def _object_metadata_from_row(row, mask_path, object_type):
    return {
        **_screen_row_meta(row),
        "file": str(mask_path),
        "object_type": object_type,
    }


def measure_body_mask(mask_path, min_area_px, max_area_px=float("inf")):
    """Measure body area as the sum of kept connected components (foreground == 1)."""
    arr = tiff.imread(str(mask_path))
    records = _measure_connected_components(
        arr == 1, min_area_px=min_area_px, max_area_px=max_area_px
    )
    raw_area_px = float(sum(r["area_px"] for r in records))
    kept_records = [r for r in records if r["kept_after_area_filter"]]
    kept_area_px = float(sum(r["area_px"] for r in kept_records))
    excluded = len(kept_records) == 0
    return kept_area_px, excluded, raw_area_px


def _measure_body_one_row(row, min_area_px, max_area_px, path_column, tag):
    mask_path = row[path_column]
    kept_area_px, excluded, raw_area_px = measure_body_mask(
        mask_path, min_area_px=min_area_px, max_area_px=max_area_px
    )
    return {
        "file": str(mask_path),
        **_screen_row_meta(row),
        f"{tag}_area_px": kept_area_px,
        f"{tag}_area_px_raw": raw_area_px,
        f"{tag}_excluded_small": bool(excluded),
    }


def compute_body_measurements(
    filemap_df,
    min_area_px,
    path_column,
    tag,
    n_jobs=1,
    max_area_px=float("inf"),
):
    """Compute body area measurements for each segmentation mask."""
    rows = list(filemap_df.to_dicts())

    if n_jobs != 1 and len(rows) > 1:
        with parallel_config(backend="loky", n_jobs=n_jobs):
            records = Parallel()(
                delayed(_measure_body_one_row)(
                    row, min_area_px, max_area_px, path_column, tag
                )
                for row in rows
            )
    else:
        records = [
            _measure_body_one_row(row, min_area_px, max_area_px, path_column, tag)
            for row in rows
        ]

    body_df = pl.DataFrame(records)
    if body_df.height > 0:
        body_df = body_df.sort(
            _sort_keys(
                body_df,
                "strain", "day", "plate", "row384", "col384", "well96", "file",
            )
        )

    return body_df


def _measure_body_objects_one_row(
    row, path_column, min_area_px, max_area_px, um2_per_px2
):
    mask_path = row[path_column]
    arr = tiff.imread(str(mask_path))
    records = _measure_connected_components(
        arr == 1, min_area_px=min_area_px, max_area_px=max_area_px
    )
    meta = _object_metadata_from_row(row, mask_path, object_type="body")
    out = []
    for record in records:
        out.append(
            {
                **meta,
                **record,
                "area_um2": float(record["area_px"] * um2_per_px2),
            }
        )
    return out


def compute_body_object_measurements(
    filemap_df,
    path_column,
    min_area_px,
    um2_per_px2,
    n_jobs=1,
    max_area_px=float("inf"),
):
    """Compute one row per segmented body object."""
    rows = list(filemap_df.to_dicts())

    if n_jobs != 1 and len(rows) > 1:
        with parallel_config(backend="loky", n_jobs=n_jobs):
            nested_records = Parallel()(
                delayed(_measure_body_objects_one_row)(
                    row, path_column, min_area_px, max_area_px, um2_per_px2
                )
                for row in rows
            )
    else:
        nested_records = [
            _measure_body_objects_one_row(
                row, path_column, min_area_px, max_area_px, um2_per_px2
            )
            for row in rows
        ]

    records = [record for row_records in nested_records for record in row_records]
    body_obj_df = pl.DataFrame(records) if records else pl.DataFrame()
    if body_obj_df.height > 0:
        body_obj_df = body_obj_df.sort(
            _sort_keys(
                body_obj_df,
                "strain", "day", "plate", "row384", "col384", "file", "object_label",
            )
        )
    return body_obj_df


def measure_pharynx_mask(mask_path, min_area_px, max_area_px=float("inf")):
    """Measure pharynx objects from a segmentation mask (foreground == 1)."""
    arr = tiff.imread(str(mask_path))
    ph = arr == 1

    if not ph.any():
        return 0, 0.0, np.nan

    structure = np.ones((3, 3), dtype=bool)
    lbl, n = ndi.label(ph, structure=structure)
    if n == 0:
        return 0, 0.0, np.nan

    areas = np.bincount(lbl.ravel())[1:]
    areas = areas[(areas >= min_area_px) & (areas <= max_area_px)]

    if areas.size == 0:
        return 0, 0.0, np.nan

    return int(areas.size), float(areas.sum()), float(areas.mean())


def _measure_pharynx_one_row(row, pharynx_col, min_area_px, max_area_px, tag):
    mask_path = row[pharynx_col]
    ph_count, ph_area_sum, ph_area_mean = measure_pharynx_mask(
        mask_path, min_area_px=min_area_px, max_area_px=max_area_px
    )
    return {
        "file": str(mask_path),
        **_screen_row_meta(row),
        f"{tag}_count": ph_count,
        f"{tag}_area_sum": ph_area_sum,
        f"{tag}_area_mean": ph_area_mean,
    }


def compute_pharynx_measurements(
    filemap_df,
    pharynx_col,
    min_area_px,
    tag,
    n_jobs=1,
    max_area_px=float("inf"),
):
    """Compute pharynx measurements for each segmentation mask."""
    rows = list(filemap_df.to_dicts())

    if n_jobs != 1 and len(rows) > 1:
        with parallel_config(backend="loky", n_jobs=n_jobs):
            records = Parallel()(
                delayed(_measure_pharynx_one_row)(
                    row, pharynx_col, min_area_px, max_area_px, tag
                )
                for row in rows
            )
    else:
        records = [
            _measure_pharynx_one_row(row, pharynx_col, min_area_px, max_area_px, tag)
            for row in rows
        ]

    ph_df = pl.DataFrame(records)
    if ph_df.height > 0:
        ph_df = ph_df.sort(
            _sort_keys(
                ph_df,
                "strain", "day", "plate", "row384", "col384", "well96", "file",
            )
        )

    return ph_df


def _measure_pharynx_objects_one_row(
    row, pharynx_col, min_area_px, max_area_px, um2_per_px2
):
    mask_path = row[pharynx_col]
    arr = tiff.imread(str(mask_path))
    records = _measure_connected_components(
        arr == 1, min_area_px=min_area_px, max_area_px=max_area_px
    )
    meta = _object_metadata_from_row(row, mask_path, object_type="pharynx")
    out = []
    for record in records:
        out.append(
            {
                **meta,
                **record,
                "area_um2": float(record["area_px"] * um2_per_px2),
            }
        )
    return out


def compute_pharynx_object_measurements(
    filemap_df,
    pharynx_col,
    min_area_px,
    um2_per_px2,
    n_jobs=1,
    max_area_px=float("inf"),
):
    """Compute one row per segmented pharynx object."""
    rows = list(filemap_df.to_dicts())

    if n_jobs != 1 and len(rows) > 1:
        with parallel_config(backend="loky", n_jobs=n_jobs):
            nested_records = Parallel()(
                delayed(_measure_pharynx_objects_one_row)(
                    row, pharynx_col, min_area_px, max_area_px, um2_per_px2
                )
                for row in rows
            )
    else:
        nested_records = [
            _measure_pharynx_objects_one_row(
                row, pharynx_col, min_area_px, max_area_px, um2_per_px2
            )
            for row in rows
        ]

    records = [record for row_records in nested_records for record in row_records]
    ph_obj_df = pl.DataFrame(records) if records else pl.DataFrame()
    if ph_obj_df.height > 0:
        ph_obj_df = ph_obj_df.sort(
            _sort_keys(
                ph_obj_df,
                "strain", "day", "plate", "row384", "col384", "file", "object_label",
            )
        )
    return ph_obj_df


def _object_csv_filename(first_row, channel_tag):
    plate = str(first_row.get("plate", "plate")).replace("/", "_")
    row384 = str(first_row.get("row384", "NA"))
    col384 = int(first_row.get("col384", 0) or 0)
    return (
        f"{channel_tag}_objects_"
        f"{first_row.get('strain', 'NA')}_day{first_row.get('day', 'NA')}_"
        f"{plate}_{row384}{col384:02d}.csv"
    )


def write_per_mask_object_csvs(object_df, output_dir, channel_tag):
    """Write one object CSV per mask/image and return a file->csv mapping."""
    channel_tag = extract_primary_channel_tag(channel_tag)
    object_csv_col = f"{channel_tag}_object_csv"
    os.makedirs(output_dir, exist_ok=True)

    if object_df is None or object_df.height == 0:
        return pl.DataFrame(
            schema={
                "file": pl.String,
                object_csv_col: pl.String,
            }
        )

    records = []
    for part in object_df.partition_by("file", maintain_order=True):
        first_row = part.row(0, named=True)
        filename = _object_csv_filename(first_row, channel_tag)
        csv_path = os.path.join(output_dir, filename)
        part.write_csv(csv_path)
        records.append(
            {
                "file": str(first_row["file"]),
                object_csv_col: csv_path,
            }
        )

    return pl.DataFrame(records).sort("file")


def build_well_level_table(body_df, pharynx_df, um2_per_px2, body_tag, pharynx_tag):
    """Merge body and pharynx measurements and aggregate to well level."""
    # Join on well position only; gene columns may differ in dtype between channels.
    position_keys = _well_position_key_cols(body_df, pharynx_df)
    gene_cols = [c for c in GENE_ANNOTATION_COLS if c in body_df.columns]
    body_group_keys = position_keys + gene_cols

    bt, pt = body_tag, pharynx_tag

    body_agg = body_df.group_by(body_group_keys, maintain_order=True).agg(
        pl.col(f"{bt}_area_px").sum().alias(f"{bt}_total_area_px"),
        pl.col(f"{bt}_area_px_raw").sum().alias(f"{bt}_area_px_raw"),
        pl.col(f"{bt}_excluded_small").sum().alias(f"{bt}_n_excluded_small"),
    )

    pharynx_agg = pharynx_df.group_by(position_keys, maintain_order=True).agg(
        pl.col(f"{pt}_count").sum().alias(f"{pt}_total_count"),
        pl.col(f"{pt}_area_sum").sum().alias(f"{pt}_area_sum"),
        pl.col(f"{pt}_area_mean").mean().alias(f"{pt}_area_mean"),
    )

    well_df = body_agg.join(pharynx_agg, on=position_keys, how="left")

    # Keep wells that have pharynx measurements but no body mask.
    missing_from_body = pharynx_agg.join(body_agg, on=position_keys, how="anti")
    if missing_from_body.height > 0:
        missing_from_body = missing_from_body.with_columns(
            pl.lit(None, dtype=well_df.schema[f"{bt}_total_area_px"]).alias(
                f"{bt}_total_area_px"
            ),
            pl.lit(None, dtype=well_df.schema[f"{bt}_area_px_raw"]).alias(
                f"{bt}_area_px_raw"
            ),
            pl.lit(None, dtype=well_df.schema[f"{bt}_n_excluded_small"]).alias(
                f"{bt}_n_excluded_small"
            ),
        )
        missing_from_body = missing_from_body.select(well_df.columns)
        well_df = pl.concat([well_df, missing_from_body], how="vertical", rechunk=True)

    well_df = well_df.with_columns(
        pl.col(f"{bt}_total_area_px").fill_null(0.0),
        pl.col(f"{bt}_area_px_raw").fill_null(0.0),
        pl.col(f"{bt}_n_excluded_small").fill_null(0).cast(pl.Int64),
        pl.col(f"{pt}_total_count").fill_null(0).cast(pl.Int64),
        pl.col(f"{pt}_area_sum").fill_null(0.0),
    )

    well_df = well_df.with_columns(
        (pl.col(f"{bt}_total_area_px") * um2_per_px2).alias(f"{bt}_total_area_um2"),
        (pl.col(f"{pt}_area_sum") * um2_per_px2).alias(f"{pt}_area_sum_um2"),
    )

    sort_keys = [
        c for c in ["strain", "day", "plate", "row384", "col384"] if c in well_df.columns
    ]
    well_df = well_df.sort(sort_keys)

    return well_df


def build_gene_level_table(well_df, body_tag, pharynx_tag):
    """Aggregate well-level measurements to gene level (one row per gene per plate)."""
    key_cols = [c for c in GENE_LEVEL_KEY_COLS if c in well_df.columns]
    if not key_cols:
        raise ValueError("well_df is missing columns required for gene-level aggregation")

    bt, pt = body_tag, pharynx_tag

    n_wells_expr = (
        pl.col("well96").n_unique().alias("n_wells")
        if "well96" in well_df.columns
        else pl.len().alias("n_wells")
    )

    gene_df = well_df.group_by(key_cols, maintain_order=True).agg(
        n_wells_expr,
        pl.col(f"{pt}_total_count").sum().alias(f"{pt}_total_count"),
        pl.col(f"{bt}_total_area_px").sum().alias(f"{bt}_total_area_px"),
        pl.col(f"{bt}_total_area_um2").sum().alias(f"{bt}_total_area_um2"),
        pl.col(f"{bt}_area_px_raw").sum().alias(f"{bt}_area_px_raw"),
        pl.col(f"{bt}_n_excluded_small").sum().alias(f"{bt}_n_excluded_small"),
    )

    gene_df = gene_df.with_columns(
        pl.when(pl.col(f"{pt}_total_count") > 0)
        .then(pl.col(f"{bt}_total_area_px") / pl.col(f"{pt}_total_count"))
        .otherwise(None)
        .alias(f"avg_{bt}_per_{pt}_total_count_px"),
        pl.when(pl.col(f"{pt}_total_count") > 0)
        .then(pl.col(f"{bt}_total_area_um2") / pl.col(f"{pt}_total_count"))
        .otherwise(None)
        .alias(f"avg_{bt}_per_{pt}_total_count_um2"),
    )

    sort_keys = [
        c for c in ["strain", "day", "plate", "gene_name"] if c in gene_df.columns
    ]
    gene_df = gene_df.sort(sort_keys)

    return gene_df


def build_gene_qc_table(qc_image_df, body_tag, pharynx_tag):
    """Aggregate per-image QC stats to gene level."""
    if qc_image_df is None:
        return None

    bt, pt = body_tag, pharynx_tag
    gene_key_cols = [c for c in GENE_LEVEL_KEY_COLS if c in qc_image_df.columns]
    gene_qc = (
        qc_image_df.group_by(gene_key_cols, maintain_order=True)
        .agg(
            pl.col(f"{pt}_total_count_qc").sum().alias(f"{pt}_total_count_qc"),
            pl.col(f"{bt}_total_area_px_qc").sum().alias(f"{bt}_total_area_px_qc"),
            pl.col(f"{bt}_total_area_um2_qc").sum().alias(f"{bt}_total_area_um2_qc"),
        )
        .with_columns(
            pl.when(pl.col(f"{pt}_total_count_qc") > 0)
            .then(pl.col(f"{bt}_total_area_px_qc") / pl.col(f"{pt}_total_count_qc"))
            .otherwise(None)
            .alias(f"avg_{bt}_per_{pt}_total_count_px_qc"),
            pl.when(pl.col(f"{pt}_total_count_qc") > 0)
            .then(pl.col(f"{bt}_total_area_um2_qc") / pl.col(f"{pt}_total_count_qc"))
            .otherwise(None)
            .alias(f"avg_{bt}_per_{pt}_total_count_um2_qc"),
        )
    )
    qc_nonkey = [c for c in gene_qc.columns if c not in set(gene_key_cols)]
    return gene_qc.rename({c: f"{c}_gene" for c in qc_nonkey})


def build_combined_screen_table(
    body_df,
    pharynx_meas_df,
    well_df,
    gene_df,
    body_tag,
    pharynx_tag,
    body_object_paths_df=None,
    pharynx_object_paths_df=None,
):
    """Build combined table with per-image, well- and gene-level info."""
    if body_df is None:
        return None

    bt, pt = body_tag, pharynx_tag

    combined = body_df.rename({"file": f"{bt}_file"})

    if body_object_paths_df is not None and body_object_paths_df.height > 0:
        body_object_paths = body_object_paths_df.rename({"file": f"{bt}_file"})
        combined = combined.join(body_object_paths, on=f"{bt}_file", how="left")
        body_object_col = body_object_paths.columns[-1]
        if body_object_col in combined.columns:
            cols_reordered = [c for c in combined.columns if c != body_object_col]
            bt_pos = cols_reordered.index(f"{bt}_file")
            cols_reordered.insert(bt_pos + 1, body_object_col)
            combined = combined.select(cols_reordered)

    if pharynx_meas_df is not None:
        if "file" in pharynx_meas_df.columns and pt:
            pharynx_stats = pharynx_meas_df.rename({"file": f"{pt}_file"})
        else:
            pharynx_stats = pharynx_meas_df
        key_cols = _well_position_key_cols(combined, pharynx_stats)
        combined = combined.join(pharynx_stats, on=key_cols, how="left")


        pt_file_col = f"{pt}_file"
        bt_file_col = f"{bt}_file"
        if pt_file_col in combined.columns and bt_file_col in combined.columns:
            cols_reordered = [c for c in combined.columns if c != pt_file_col]
            bt_pos = cols_reordered.index(bt_file_col)
            cols_reordered.insert(bt_pos + 1, pt_file_col)
            combined = combined.select(cols_reordered)

    if pharynx_object_paths_df is not None and pharynx_object_paths_df.height > 0 and pt:
        pharynx_object_paths = pharynx_object_paths_df.rename({"file": f"{pt}_file"})
        combined = combined.join(pharynx_object_paths, on=f"{pt}_file", how="left")
        pharynx_object_col = pharynx_object_paths.columns[-1]
        if pharynx_object_col in combined.columns and f"{pt}_file" in combined.columns:
            cols_reordered = [c for c in combined.columns if c != pharynx_object_col]
            pt_pos = cols_reordered.index(f"{pt}_file")
            cols_reordered.insert(pt_pos + 1, pharynx_object_col)
            combined = combined.select(cols_reordered)

    if well_df is not None:
        well_key_cols = _join_key_cols(
            combined,
            well_df,
            base_cols=[
                "strain", "day", "plate", "row384", "col384", "well96",
                "lib_code", "gene_name", "is_control",
            ],
        )
        # Suffix well-level columns to avoid name clashes with per-image columns.
        well_renamed = well_df.rename(
            {
                f"{bt}_area_px_raw": f"{bt}_area_px_raw_well",
                f"{pt}_area_sum": f"{pt}_area_sum_well",
                f"{pt}_area_sum_um2": f"{pt}_area_sum_um2_well",
            }
        )
        combined = combined.join(well_renamed, on=well_key_cols, how="left")
        for extra in ("row384_right", "col384_right"):
            if extra in combined.columns:
                combined = combined.drop(extra)

    if gene_df is not None:
        gene_key_cols = [c for c in GENE_LEVEL_KEY_COLS if c in gene_df.columns]
        # Suffix gene-level columns to avoid name clashes with per-image columns.
        gene_renamed = gene_df.rename(
            {
                f"{pt}_total_count": f"{pt}_total_count_gene",
                f"{bt}_total_area_px": f"{bt}_total_area_px_gene",
                f"{bt}_total_area_um2": f"{bt}_total_area_um2_gene",
                f"{bt}_area_px_raw": f"{bt}_area_px_raw_gene",
                f"{bt}_n_excluded_small": f"{bt}_n_excluded_small_gene",
            }
        )
        combined = combined.join(gene_renamed, on=gene_key_cols, how="left")

    return combined
    