"""Compute body/pharynx morphology tables for RNAi screen experiments."""

import logging
import os

import polars as pl
import utils
from towbintools.foundation.file_handling import write_filemap

from screen_measurements import (
    DEFAULT_MAX_BODY_AREA_PX,
    DEFAULT_MAX_PHARYNX_AREA_PX,
    DEFAULT_MIN_BODY_AREA_PX,
    DEFAULT_MIN_PHARYNX_AREA_PX,
    build_combined_screen_table,
    build_gene_level_table,
    build_gene_qc_table,
    build_well_level_table,
    compute_body_measurements,
    compute_body_object_measurements,
    compute_pharynx_measurements,
    compute_pharynx_object_measurements,
    extract_channel_tag,
    extract_primary_channel_tag,
    write_per_mask_object_csvs,
)
from screen_utils import (
    apply_good_vs_error_qc,
    build_annotated_screen_filemap,
    build_qc_image_level_stats,
    filemap_has_gene_labels,
    resolve_experiment_root,
)

logging.basicConfig(level=logging.INFO)


def _write_screen_table(df: pl.DataFrame, path: str | None) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_filemap(df, path)


def _load_mask_root(input_pickle: str):
    if input_pickle.endswith(".pkl"):
        root = utils.load_pickles(input_pickle)[0]
        if isinstance(root, (list, tuple)):
            root = root[0]
        return str(root)
    return str(input_pickle)


def main(input_pickle, output_file, config, n_jobs):
    """Build screen filemap and well/gene/combined morphology CSVs."""
    config = utils.load_pickles(config)[0]

    body_root = _load_mask_root(input_pickle)
    experiment_root = resolve_experiment_root(body_root)
    experiment_dir = config.get("experiment_dir") or experiment_root

    pattern = config.get("pattern", "*.tif*")
    analysis_col = config.get("analysis_col", "analysis/ch2_seg")
    exp_folder_regex = config.get("exp_folder_regex")
    min_body_area_px = config.get("min_body_area_px", DEFAULT_MIN_BODY_AREA_PX)
    max_body_area_px = config.get("max_body_area_px", DEFAULT_MAX_BODY_AREA_PX)
    min_pharynx_area_px = config.get("min_pharynx_area_px", DEFAULT_MIN_PHARYNX_AREA_PX)
    max_pharynx_area_px = config.get("max_pharynx_area_px", DEFAULT_MAX_PHARYNX_AREA_PX)
    qc_max_pair_distance_px = float(config.get("qc_max_pair_distance_px", 2000.0))

    um_per_px = config.get("pixelsize") or config.get("um_per_px") or (6.5 / 4.0)
    um2_per_px2 = float(um_per_px) ** 2

    pharynx_root = config.get("pharynx_root")
    enable_qc_good_vs_error = bool(config.get("enable_qc_good_vs_error", False))
    enable_object_level = bool(
        config.get("enable_object_level_measurements", enable_qc_good_vs_error)
    )
    qc_models_dir = config.get("qc_models_dir", "")
    body_raw_channel = int(config.get("body_raw_channel", 1))
    pharynx_raw_channel = int(config.get("pharynx_raw_channel", 0))

    body_tag = extract_channel_tag(analysis_col)
    pharynx_tag = extract_channel_tag(pharynx_root) if pharynx_root else None
    logging.info("Channel tags: body=%s, pharynx=%s", body_tag, pharynx_tag)
    if enable_object_level:
        logging.info(
            "Object-level measurements: enabled (QC classifier=%s)",
            "on" if enable_qc_good_vs_error and qc_models_dir else "off",
        )
    else:
        logging.info("Object-level measurements: disabled")

    out_dir = os.path.dirname(output_file)
    body_object_dir = (
        config.get(
            "body_object_dir",
            os.path.join(out_dir, f"{extract_primary_channel_tag(body_tag)}_object_csvs"),
        )
        if enable_object_level
        else None
    )
    pharynx_object_dir = (
        config.get(
            "pharynx_object_dir",
            os.path.join(
                out_dir, f"{extract_primary_channel_tag(pharynx_tag)}_object_csvs"
            ),
        )
        if enable_object_level and pharynx_tag
        else None
    )
    well_output = config.get(
        "well_output",
        os.path.join(out_dir, f"{body_tag}_{pharynx_tag}_well_level.csv")
        if pharynx_tag
        else os.path.join(out_dir, f"{body_tag}_well_level.csv"),
    )
    gene_output = config.get(
        "gene_output",
        os.path.join(out_dir, f"{body_tag}_{pharynx_tag}_gene_level.csv")
        if pharynx_tag
        else os.path.join(out_dir, f"{body_tag}_gene_level.csv"),
    )
    combined_output = config.get(
        "combined_output",
        os.path.join(out_dir, f"{body_tag}_{pharynx_tag}_combined.csv")
        if pharynx_tag
        else os.path.join(out_dir, f"{body_tag}_combined.csv"),
    )

    require_gene_annotation = config.get("require_gene_annotation")

    # Annotated filemap: mask paths + plate/well metadata (+ genes when available).
    df = build_annotated_screen_filemap(
        body_root,
        pattern=pattern,
        analysis_col=analysis_col,
        exp_folder_regex=exp_folder_regex,
        experiment_root=experiment_dir,
        plate_annotation_dir=config.get("plate_annotation_dir"),
        plate_annotation_format=config.get("plate_annotation_format", "auto"),
        plate_name_map=config.get("plate_name_map"),
        require_gene_annotation=require_gene_annotation,
    )
    has_gene_labels = filemap_has_gene_labels(df)
    if not has_gene_labels and gene_output:
        logging.info(
            "No gene annotations available; skipping gene-level output."
        )
        gene_output = None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.write_csv(output_file)
    logging.info("Saved filemap with %s rows to %s", df.height, output_file)

    body_obj_df = None
    pharynx_obj_df = None
    body_object_paths_df = None
    pharynx_object_paths_df = None

    # Per-image body measurements (and optional object-level rows).
    body_df = compute_body_measurements(
        df,
        min_body_area_px,
        analysis_col,
        tag=body_tag,
        n_jobs=n_jobs,
        max_area_px=max_body_area_px,
    )
    logging.info("Computed body measurements with %s rows", body_df.height)

    if enable_object_level and body_object_dir:
        # One row per connected component; optional good_vs_error classifier.
        body_obj_df = compute_body_object_measurements(
            df,
            path_column=analysis_col,
            min_area_px=min_body_area_px,
            um2_per_px2=um2_per_px2,
            n_jobs=n_jobs,
            max_area_px=max_body_area_px,
        )
        if enable_qc_good_vs_error and qc_models_dir:
            logging.info("Applying body good_vs_error QC inference to object rows")
            body_obj_df = apply_good_vs_error_qc(
                body_obj_df,
                channel_name="body",
                models_dir=qc_models_dir,
                raw_channel=body_raw_channel,
                experiment_root=experiment_root,
                mask_positive_mode="eq1",
            )
        body_object_paths_df = write_per_mask_object_csvs(
            body_obj_df,
            output_dir=body_object_dir,
            channel_tag=body_tag,
        )
        logging.info(
            "Saved body object CSV paths for %s masks under %s",
            body_object_paths_df.height,
            body_object_dir,
        )

    pharynx_meas_df = None
    if pharynx_root:
        # Same annotated filemap as body so well-position joins match reliably.
        pharynx_df = build_annotated_screen_filemap(
            pharynx_root,
            pattern=pattern,
            analysis_col=analysis_col,
            exp_folder_regex=exp_folder_regex,
            experiment_root=experiment_dir,
            plate_annotation_dir=config.get("plate_annotation_dir"),
            plate_annotation_format=config.get("plate_annotation_format", "auto"),
            plate_name_map=config.get("plate_name_map"),
            require_gene_annotation=require_gene_annotation,
        )
        pharynx_meas_df = compute_pharynx_measurements(
            pharynx_df,
            pharynx_col=analysis_col,
            min_area_px=min_pharynx_area_px,
            tag=pharynx_tag,
            n_jobs=n_jobs,
            max_area_px=max_pharynx_area_px,
        )
        logging.info("Computed pharynx measurements with %s rows", pharynx_meas_df.height)

        if enable_object_level and pharynx_object_dir:
            pharynx_obj_df = compute_pharynx_object_measurements(
                pharynx_df,
                pharynx_col=analysis_col,
                min_area_px=min_pharynx_area_px,
                um2_per_px2=um2_per_px2,
                n_jobs=n_jobs,
                max_area_px=max_pharynx_area_px,
            )
            if enable_qc_good_vs_error and qc_models_dir:
                logging.info("Applying pharynx good_vs_error QC inference to object rows")
                pharynx_obj_df = apply_good_vs_error_qc(
                    pharynx_obj_df,
                    channel_name="pharynx",
                    models_dir=qc_models_dir,
                    raw_channel=pharynx_raw_channel,
                    experiment_root=experiment_root,
                    mask_positive_mode="eq1",
                )
            pharynx_object_paths_df = write_per_mask_object_csvs(
                pharynx_obj_df,
                output_dir=pharynx_object_dir,
                channel_tag=pharynx_tag,
            )
            logging.info(
                "Saved pharynx object CSV paths for %s masks under %s",
                pharynx_object_paths_df.height,
                pharynx_object_dir,
            )

    qc_image_df = None
    if (
        enable_object_level
        and enable_qc_good_vs_error
        and body_obj_df is not None
        and pharynx_obj_df is not None
    ):
        # QC-adjusted image stats from paired body/pharynx objects.
        qc_image_df = build_qc_image_level_stats(
            body_obj_df,
            pharynx_obj_df,
            body_tag=body_tag,
            pharynx_tag=pharynx_tag,
            um2_per_px2=um2_per_px2,
            max_pair_distance_px=qc_max_pair_distance_px,
        )
        if qc_image_df is not None:
            logging.info(
                "Built QC-adjusted image stats with %s rows", qc_image_df.height
            )

    well_df = None
    gene_df = None
    if well_output:
        # Well- and gene-level aggregation; optional QC columns joined on well keys.
        well_df = build_well_level_table(
            body_df,
            pharynx_meas_df,
            um2_per_px2,
            body_tag=body_tag,
            pharynx_tag=pharynx_tag,
        )
        if qc_image_df is not None:
            well_key_cols = [
                c
                for c in [
                    "strain",
                    "day",
                    "plate",
                    "row384",
                    "col384",
                    "well96",
                    "lib_code",
                    "gene_name",
                    "is_control",
                ]
                if c in well_df.columns and c in qc_image_df.columns
            ]
            well_df = well_df.join(qc_image_df, on=well_key_cols, how="left")

        _write_screen_table(well_df, well_output)
        logging.info(
            "Saved well-level measurements with %s rows to %s",
            well_df.height,
            well_output,
        )

    if gene_output:
        if well_df is None:
            raise ValueError("gene_output in config requires well_output to be set")
        gene_df = build_gene_level_table(
            well_df, body_tag=body_tag, pharynx_tag=pharynx_tag
        )
        gene_qc = build_gene_qc_table(qc_image_df, body_tag, pharynx_tag)
        if gene_qc is not None:
            gene_key_cols = [c for c in gene_df.columns if c in gene_qc.columns]
            gene_df = gene_df.join(gene_qc, on=gene_key_cols, how="left")

        _write_screen_table(gene_df, gene_output)
        logging.info(
            "Saved gene-level measurements with %s rows to %s",
            gene_df.height,
            gene_output,
        )

    if combined_output:
        combined = build_combined_screen_table(
            body_df,
            pharynx_meas_df,
            well_df,
            gene_df,
            body_tag=body_tag,
            pharynx_tag=pharynx_tag,
            body_object_paths_df=body_object_paths_df,
            pharynx_object_paths_df=pharynx_object_paths_df,
        )
        if combined is not None:
            _write_screen_table(combined, combined_output)
            logging.info(
                "Saved combined screen measurements with %s rows to %s",
                combined.height,
                combined_output,
            )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
