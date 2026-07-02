"""Helpers for RNAi screen filemaps, plate annotation, and optional QC."""

import logging
import re
from pathlib import Path

import numpy as np
import polars as pl
import tifffile as tiff
import xgboost as xgb
from joblib import load as joblib_load
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from towbintools.classification.qc_tools import compute_qc_features


DEFAULT_EXP_FOLDER_REGEX = [
    # Override via config key exp_folder_regex.
    r"^(?P<strain>\d+)_day(?P<day>\d+)_(?P<plate>[^_]+)_?",
    r"^(?P<date>\d{8})_wBT(?P<strain>\d+)_(?P<plate>[^_]+)$",
]

WELL_RE = re.compile(r"Well([A-P])(\d{2})", re.IGNORECASE)


def normalize_exp_folder_regex(patterns) -> list[str]:
    """Normalize config value to a list of regex pattern strings."""
    if patterns is None:
        return list(DEFAULT_EXP_FOLDER_REGEX)
    if isinstance(patterns, str):
        p = patterns.strip()
        return [p] if p else list(DEFAULT_EXP_FOLDER_REGEX)
    out = [str(p).strip() for p in patterns if str(p).strip()]
    return out or list(DEFAULT_EXP_FOLDER_REGEX)


def compile_exp_folder_patterns(patterns=None) -> list[re.Pattern]:
    """Compile plate-folder regexes (tried in order; first match wins)."""
    return [
        re.compile(p, re.IGNORECASE)
        for p in normalize_exp_folder_regex(patterns)
    ]


def _exp_info_from_match(m: re.Match) -> dict | None:
    gd = m.groupdict()
    strain = gd.get("strain")
    plate = gd.get("plate")
    day_raw = gd.get("day")
    if plate is None or str(plate).strip() == "":
        return None
    # Strain is optional (e.g. Control_plate folders without wBT<number>).
    if strain is None or str(strain).strip() == "":
        strain = "0"
    info: dict = {
        "strain": int(strain),
        "plate": str(plate),
    }
    # Only the optional ``day`` group is stored; ``date`` is ignored.
    if day_raw is not None and str(day_raw).strip() != "":
        info["day"] = int(day_raw)
    return info


def parse_exp_info(exp_dir: str | Path, exp_folder_regex=None):
    """Parse plate folder name into strain, plate, and optional day."""
    name = Path(exp_dir).name
    for pat in compile_exp_folder_patterns(exp_folder_regex):
        m = pat.match(name)
        if m:
            info = _exp_info_from_match(m)
            if info is not None:
                return info
    return None


def is_exp_plate_folder(folder_name: str, exp_folder_regex=None) -> bool:
    """True if ``folder_name`` matches any configured plate-folder regex."""
    return parse_exp_info(folder_name, exp_folder_regex=exp_folder_regex) is not None


def find_exp_plate_subfolder_in_parts(
    parts: tuple[str, ...] | list[str],
    exp_folder_regex=None,
) -> str | None:
    """Return the first path component that matches a plate-folder regex."""
    for pt in parts:
        if is_exp_plate_folder(str(pt), exp_folder_regex=exp_folder_regex):
            return str(pt)
    return None

ROWS_384 = list("ABCDEFGHIJKLMNOP")
ROWS_96 = list("ABCDEFGH")


def resolve_experiment_root(path_like: str) -> str:
    """Return the nearest parent directory that contains a ``raw/`` folder."""
    path = Path(str(path_like)).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "raw").exists():
            return str(candidate)
    return str(path)


def mask_path_to_raw(mask_path: str, experiment_dir: str) -> Path | None:
    """Map mask TIFF path to raw TIFF path under <experiment_dir>/raw/<subfolder>/..."""
    try:
        if not experiment_dir:
            return None
        p = Path(mask_path)
        exp = Path(experiment_dir)
        rel = p.relative_to(exp)
        subfolder = find_exp_plate_subfolder_in_parts(rel.parts)
        if subfolder is None:
            return None
        cand = exp / "raw" / subfolder / p.name
        return cand if cand.exists() else None
    except Exception:
        return None


def read_raw_channel(raw_path: Path, channel: int) -> np.ndarray:
    with tiff.TiffFile(str(raw_path)) as tf:
        n_pages = len(tf.pages)
        if n_pages > 1:
            pg = min(channel, n_pages - 1)
            arr = tf.pages[pg].asarray()
        else:
            arr = tf.pages[0].asarray()
            if arr.ndim == 3:
                ch = min(channel, arr.shape[2] - 1)
                arr = arr[:, :, ch]
    return arr


def _load_good_vs_error_model(models_dir: str, channel: str):
    pkl_path = Path(models_dir) / f"qc_xgb_{channel}_good_vs_error.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing model bundle: {pkl_path}")
    bundle = joblib_load(pkl_path)
    model = xgb.XGBClassifier()
    model.load_model(bundle["model_path"])
    return model, bundle["feature_columns"]


def apply_good_vs_error_qc(
    object_df: pl.DataFrame,
    *,
    channel_name: str,
    models_dir: str,
    raw_channel: int,
    experiment_root: str,
    mask_positive_mode: str,
) -> pl.DataFrame:
    """Add a single pred_class column to object-level table."""
    if object_df is None or object_df.height == 0:
        return object_df
    if not models_dir:
        return object_df

    model, feature_cols = _load_good_vs_error_model(models_dir, channel_name)
    rows = object_df.to_dicts()

    last_mask_path = None
    last_lbl = None
    last_raw_key = None
    last_raw = None

    preds = []

    for row in rows:
        try:
            mask_path = Path(str(row["file"]))
            if not mask_path.exists():
                preds.append(None)
                continue

            if str(mask_path) != last_mask_path:
                mask_arr = tiff.imread(str(mask_path))
                if mask_positive_mode == "eq1":
                    fg = mask_arr == 1
                else:
                    fg = mask_arr > 0
                lbl, _n = ndi.label(fg, structure=np.ones((3, 3), dtype=bool))
                last_lbl = lbl
                last_mask_path = str(mask_path)
            else:
                lbl = last_lbl

            comp_id = int(row["object_label"])
            comp = lbl == comp_id
            if comp.sum() == 0:
                preds.append(None)
                continue

            y0, y1 = int(row["bbox_min_row"]), int(row["bbox_max_row"])
            x0, x1 = int(row["bbox_min_col"]), int(row["bbox_max_col"])

            raw_path = mask_path_to_raw(str(mask_path), experiment_root)
            if raw_path is None or not raw_path.exists():
                preds.append(None)
                continue

            raw_key = (str(raw_path), int(raw_channel))
            if raw_key != last_raw_key:
                last_raw = read_raw_channel(raw_path, channel=int(raw_channel))
                last_raw_key = raw_key
            raw = last_raw

            crop_raw = raw[y0:y1, x0:x1]
            crop_mask = comp[y0:y1, x0:x1].astype(np.uint8)

            feats = compute_qc_features(crop_mask, crop_raw)
            if feats is None or len(feats) == 0:
                preds.append(None)
                continue

            X = feats.reindex(columns=feature_cols, fill_value=0.0)
            p_err = float(model.predict_proba(X)[:, 1][0])
            pred = "error" if p_err >= 0.5 else "good"
            preds.append(pred)
        except Exception:
            preds.append(None)

    return object_df.with_columns(pl.Series("pred_class", preds))


_QC_PAIR_KEY_COLS = [
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


def _filter_objects_for_area_match(df: pl.DataFrame) -> pl.DataFrame:
    """Keep objects that pass the area filter."""
    if "kept_after_area_filter" not in df.columns:
        return df
    return df.filter(pl.col("kept_after_area_filter").fill_null(False))


def _pred_is_error(pred_class) -> bool:
    return str(pred_class or "").strip().lower() == "error"


def _finite_centroid_rows(df: pl.DataFrame) -> list[dict]:
    rows = []
    for row in df.to_dicts():
        cy, cx = row.get("centroid_y"), row.get("centroid_x")
        if cy is None or cx is None:
            continue
        if not np.isfinite(float(cy)) or not np.isfinite(float(cx)):
            continue
        rows.append(row)
    return rows


def _pair_body_pharynx_objects(
    body_rows: list[dict],
    pharynx_rows: list[dict],
    *,
    max_pair_distance_px: float,
) -> tuple[float, int]:
    """Match body and pharynx objects at one well position (1:1 by centroid)."""
    if not body_rows or not pharynx_rows:
        return 0.0, 0

    b_y = np.array([float(r["centroid_y"]) for r in body_rows], dtype=float)
    b_x = np.array([float(r["centroid_x"]) for r in body_rows], dtype=float)
    p_y = np.array([float(r["centroid_y"]) for r in pharynx_rows], dtype=float)
    p_x = np.array([float(r["centroid_x"]) for r in pharynx_rows], dtype=float)

    dist = np.hypot(b_y[:, None] - p_y[None, :], b_x[:, None] - p_x[None, :])
    row_ind, col_ind = linear_sum_assignment(dist)  # 1:1 pairing by centroid distance

    total_body_area_px = 0.0
    n_pairs = 0
    for body_idx, ph_idx in zip(row_ind, col_ind):
        if float(dist[body_idx, ph_idx]) > max_pair_distance_px:
            continue
        body_row = body_rows[body_idx]
        ph_row = pharynx_rows[ph_idx]
        # Drop the pair if either object is a QC error.
        if _pred_is_error(body_row.get("pred_class")) or _pred_is_error(ph_row.get("pred_class")):
            continue
        total_body_area_px += float(body_row.get("area_px") or 0.0)
        n_pairs += 1

    return total_body_area_px, n_pairs


def build_qc_image_level_stats(
    body_obj_df: pl.DataFrame | None,
    pharynx_obj_df: pl.DataFrame | None,
    *,
    body_tag: str,
    pharynx_tag: str,
    um2_per_px2: float,
    max_pair_distance_px: float = 2000.0,
) -> pl.DataFrame | None:
    """Build per-image QC stats from paired body/pharynx objects."""
    if body_obj_df is None or pharynx_obj_df is None:
        return None
    if body_obj_df.height == 0 or pharynx_obj_df.height == 0:
        return None

    key_cols = [
        c for c in _QC_PAIR_KEY_COLS
        if c in body_obj_df.columns and c in pharynx_obj_df.columns
    ]
    if not key_cols:
        return None

    body_area = _filter_objects_for_area_match(body_obj_df)
    pharynx_area = _filter_objects_for_area_match(pharynx_obj_df)

    def _group_key_from_row(row: dict) -> tuple:
        return tuple(row[c] for c in key_cols)

    pharynx_by_key: dict[tuple, pl.DataFrame] = {}
    for ph_part in pharynx_area.partition_by(key_cols, maintain_order=True):
        if ph_part.height == 0:
            continue
        pharynx_by_key[_group_key_from_row(ph_part.row(0, named=True))] = ph_part

    bt, pt = body_tag, pharynx_tag
    records: list[dict] = []

    for body_group in body_area.partition_by(key_cols, maintain_order=True):
        if body_group.height == 0:
            continue
        key_dict = body_group.row(0, named=True)
        key_dict = {c: key_dict[c] for c in key_cols}
        ph_group = pharynx_by_key.get(_group_key_from_row(key_dict))
        if ph_group is None or ph_group.height == 0:
            continue

        body_rows = _finite_centroid_rows(body_group)
        ph_rows = _finite_centroid_rows(ph_group)
        total_area_px, n_pharynx = _pair_body_pharynx_objects(
            body_rows,
            ph_rows,
            max_pair_distance_px=max_pair_distance_px,
        )

        records.append(
            {
                **key_dict,
                f"{bt}_total_area_px_qc": total_area_px,
                f"{pt}_total_count_qc": n_pharynx,
            }
        )

    if not records:
        return None

    qc = pl.DataFrame(records)
    qc = qc.with_columns(
        (pl.col(f"{bt}_total_area_px_qc") * float(um2_per_px2)).alias(f"{bt}_total_area_um2_qc"),
    )
    qc = qc.with_columns(
        pl.when(pl.col(f"{pt}_total_count_qc") > 0)
        .then(pl.col(f"{bt}_total_area_px_qc") / pl.col(f"{pt}_total_count_qc"))
        .otherwise(None)
        .alias(f"avg_{bt}_per_{pt}_total_count_px_qc"),
        pl.when(pl.col(f"{pt}_total_count_qc") > 0)
        .then(pl.col(f"{bt}_total_area_um2_qc") / pl.col(f"{pt}_total_count_qc"))
        .otherwise(None)
        .alias(f"avg_{bt}_per_{pt}_total_count_um2_qc"),
    )
    return qc.sort(key_cols)


def parse_well_from_filename(filename: str):
    """Parse 384‑well coordinates (row384, col384) from a filename containing 'WellA01'."""
    m = WELL_RE.search(filename)
    if not m:
        return None

    row = m.group(1).upper()
    col = int(m.group(2))
    return row, col


def well384_to_well96(row384: str, col384: int):
    """Map a 384‑well position (row384, col384) to a 96‑well (row96, col96)."""
    # Each 96-well maps from a 2x2 block of 384-well positions.
    r_idx = ROWS_384.index(row384) + 1
    c_idx = col384
    r96 = (r_idx + 1) // 2
    c96 = (c_idx + 1) // 2
    return ROWS_96[r96 - 1], int(c96)


def well96_str(row96: str, col96: int):
    """Format 96‑well coordinates as ID like 'A01'."""
    return f"{row96}{col96:02d}"


def get_dir_filemap_screen(
    root_dir,
    pattern="*.tif*",
    path_column=None,
    exp_folder_regex=None,
):
    """Build a filemap from mask TIFFs under plate subfolders."""
    if path_column is None:
        path_column = "analysis_adult_body_v2/ch2_seg"

    root_dir = Path(root_dir)
    records: list[dict] = []

    for exp_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        info = parse_exp_info(exp_dir, exp_folder_regex=exp_folder_regex)
        if info is None:
            continue

        for img_path in sorted(exp_dir.glob(pattern)):
            if not img_path.is_file():
                continue

            parsed = parse_well_from_filename(img_path.name)
            if parsed is None:
                continue

            row384, col384 = parsed
            row96, col96 = well384_to_well96(row384, col384)
            well96 = well96_str(row96, col96)

            records.append(
                {
                    path_column: str(img_path),
                    "plate": info["plate"],
                    "row384": row384,
                    "col384": col384,
                    "well96": well96,
                    "strain": info["strain"],
                    **(
                        {"day": info["day"]}
                        if info.get("day") is not None
                        else {}
                    ),
                }
            )

    df = pl.DataFrame(records)
    if df.height > 0:
        sort_keys = [
            c for c in ["strain", "day", "plate", "row384", "col384"] if c in df.columns
        ]
        df = df.sort(sort_keys)

    return df


def load_doc_plate_map(doc_csv):
    """Read an 8x12 doc plate CSV and return long‑format map."""
    doc_csv = Path(doc_csv)
    plate_name = doc_csv.stem

    df = pl.read_csv(doc_csv, separator=";", has_header=False, infer_schema_length=0)

    if df.shape != (8, 12):
        raise ValueError(f"{doc_csv} expected 8x12 but got {df.shape}")

    rows = list("ABCDEFGH")
    cols = list(range(1, 13))

    values = df.to_numpy()

    records = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            lib_code = str(values[i, j]).strip()
            records.append(
                {
                    "plate_name": plate_name,
                    "row96": r,
                    "col96": c,
                    "well96": f"{r}{c:02d}",
                    "lib_code": lib_code,
                }
            )

    return pl.DataFrame(records)


def load_rnai_library_map(rnai_csv):
    """Load rnai_library_genes.csv and build position -> gene_name mapping."""
    rnai_csv = Path(rnai_csv)
    rnai = pl.read_csv(rnai_csv, infer_schema_length=0)

    expected = {"gene_name", "Ahringer Plate Position", "Vidal Plate Position"}
    missing = expected - set(rnai.columns)
    if missing:
        raise ValueError(f"rnai_library_genes missing columns: {missing}")

    pos_to_gene: dict[str, str] = {}
    for row in rnai.iter_rows(named=True):
        gene = str(row["gene_name"]).strip()
        for col in ["Ahringer Plate Position", "Vidal Plate Position"]:
            pos = str(row[col]).strip()
            if not pos:
                continue
            if pos not in pos_to_gene:
                pos_to_gene[pos] = gene
            else:
                if pos_to_gene[pos] != gene:
                    pos_to_gene[pos] = f"{pos_to_gene[pos]}|{gene}"

    return pos_to_gene


CONTROL_ALIASES = {
    "EV": "EV",
    "empty": "empty",
    "EMPTY": "empty",
    "empty_vector": "empty",
    "EMPTY_VECTOR": "empty",
    "L4440": "empty",
    "l4440": "empty",
    "ama-1": "ama-1",
    "mex-3": "mex-3",
    "AMA-1": "ama-1",
    "MEX-3": "mex-3",
}


def canonical_control_gene_name(gene_name: str | None) -> str | None:
    """Map control naming variants to a single canonical gene_name."""
    if gene_name is None:
        return None
    x = str(gene_name).strip()
    if not x:
        return None
    if x in CONTROL_ALIASES:
        return CONTROL_ALIASES[x]
    key = x.lower().replace(" ", "_")
    alias_map = {k.lower(): v for k, v in CONTROL_ALIASES.items()}
    return alias_map.get(key, x)


def libcode_to_gene(lib_code, pos_to_gene):
    """Map a doc lib_code entry to (gene_name, is_control)."""
    if lib_code is None:
        return None, False

    x = str(lib_code).strip()
    if not x:
        return None, False

    if x in CONTROL_ALIASES:
        return CONTROL_ALIASES[x], True

    gene = pos_to_gene.get(x, None)
    if gene is not None:
        return gene, False

    x2 = re.sub(r"\s+", "", x)
    gene = pos_to_gene.get(x2, None)
    if gene is not None:
        return gene, False

    return None, False


DEFAULT_PLATE_NAME_MAP = {
    "control_plate": "Control",
    "control": "Control",
}

WELL_GENE_CONTROL_GENES = {
    "ev",
    "empty",
    "empty_vector",
    "l4440",
    "ama-1",
    "mex-3",
}


def _normalize_plate_name_map(plate_name_map) -> dict[str, str]:
    merged = dict(DEFAULT_PLATE_NAME_MAP)
    if plate_name_map:
        merged.update({str(k): str(v) for k, v in plate_name_map.items()})
    return merged


def _resolve_annotation_dir(experiment_root: Path, plate_annotation_dir) -> Path | None:
    if plate_annotation_dir in (None, ""):
        return None
    path = Path(str(plate_annotation_dir))
    if not path.is_absolute():
        path = experiment_root / path
    return path


def classify_plate_csv(csv_path: Path) -> str | None:
    """Return ``doc_grid``, ``well_gene``, or None if the file is not a plate map."""
    csv_path = Path(csv_path)
    if csv_path.name == "rnai_library_genes.csv":
        return None

    try:
        preview = pl.read_csv(csv_path, infer_schema_length=0)
        cols = {c.lower(): c for c in preview.columns}
        if "well" in cols and "gene" in cols:
            return "well_gene"
    except Exception:
        pass

    try:
        grid = pl.read_csv(
            csv_path, separator=";", has_header=False, infer_schema_length=0
        )
        if grid.shape == (8, 12):
            return "doc_grid"
    except Exception:
        pass

    return None


def load_well_gene_plate_map(plate_csv, plate_name=None):
    """Read a Well,Gene plate CSV and return a long-format annotation table."""
    plate_csv = Path(plate_csv)
    if plate_name is None:
        plate_name = plate_csv.stem

    df = pl.read_csv(plate_csv, infer_schema_length=0)
    cols = {c.lower(): c for c in df.columns}
    if "well" not in cols or "gene" not in cols:
        raise ValueError(f"{plate_csv} expected Well,Gene columns")

    well_col = cols["well"]
    gene_col = cols["gene"]

    records = []
    for row in df.iter_rows(named=True):
        well96 = str(row[well_col]).strip().upper()
        gene_name = str(row[gene_col]).strip()
        if not well96:
            continue
        if gene_name in ("", "-", "nan", "None"):
            gene_name = None
        else:
            gene_name = canonical_control_gene_name(gene_name)
        row96 = well96[0]
        col96 = int(well96[1:])
        records.append(
            {
                "plate_name": plate_name,
                "row96": row96,
                "col96": col96,
                "well96": well96,
                "lib_code": gene_name,
                "gene_name": gene_name,
                "is_control": (
                    gene_name is not None
                    and gene_name.strip().lower() in WELL_GENE_CONTROL_GENES
                ),
            }
        )

    if not records:
        raise ValueError(f"No well annotations found in {plate_csv}")

    return pl.DataFrame(records)


def _build_doc_grid_annotation(annotation_dir: Path) -> pl.DataFrame:
    doc_files = sorted(annotation_dir.glob("*.csv"))
    doc_files = [p for p in doc_files if p.name != "rnai_library_genes.csv"]
    if not doc_files:
        raise FileNotFoundError(f"No plate CSV files found in {annotation_dir}")

    preferred_stems = {"Ahr1", "Ahr2", "Ahr3Vid1", "Vid2", "Vid3"}
    filtered = [p for p in doc_files if p.stem in preferred_stems]
    if filtered:
        doc_files = filtered
    else:
        doc_files = [
            p for p in doc_files if classify_plate_csv(p) == "doc_grid"
        ]

    plate_maps = []
    for p in doc_files:
        try:
            plate_maps.append(load_doc_plate_map(p))
        except ValueError:
            continue

    if not plate_maps:
        raise ValueError(f"No 8x12 plate layout CSVs found in {annotation_dir}")

    doc_long = pl.concat(plate_maps, how="vertical")

    rnai_map_file = annotation_dir / "rnai_library_genes.csv"
    if not rnai_map_file.exists():
        raise FileNotFoundError(f"rnai_library_genes.csv not found in {annotation_dir}")

    pos_to_gene = load_rnai_library_map(rnai_map_file)
    mapping = dict(pos_to_gene)
    mapping.update(CONTROL_ALIASES)

    return doc_long.with_columns(
        gene_name=pl.col("lib_code")
        .cast(pl.Utf8)
        .str.strip_chars()
        .replace(mapping),
        is_control=pl.col("lib_code")
        .cast(pl.Utf8)
        .str.strip_chars()
        .is_in(list(CONTROL_ALIASES.keys())),
    )


def _build_well_gene_annotation(
    annotation_dir: Path, plate_name_map: dict[str, str]
) -> pl.DataFrame:
    plate_maps = []
    for csv_path in sorted(annotation_dir.glob("*.csv")):
        if classify_plate_csv(csv_path) != "well_gene":
            continue
        plate_name = plate_name_map.get(csv_path.stem, csv_path.stem)
        plate_maps.append(load_well_gene_plate_map(csv_path, plate_name=plate_name))

    if not plate_maps:
        raise ValueError(f"No Well,Gene plate CSVs found in {annotation_dir}")

    return pl.concat(plate_maps, how="vertical")


def build_plate_annotation(
    experiment_root,
    plate_annotation_dir=None,
    plate_annotation_format="auto",
    plate_name_map=None,
):
    """Build a long-format plate annotation table from doc/ or report/ CSVs."""
    experiment_root = Path(experiment_root)
    plate_name_map = _normalize_plate_name_map(plate_name_map)
    fmt = str(plate_annotation_format or "auto").strip().lower()

    resolved_dir = _resolve_annotation_dir(experiment_root, plate_annotation_dir)
    candidate_dirs = (
        [resolved_dir]
        if resolved_dir is not None
        else [experiment_root / "doc", experiment_root / "report"]
    )

    errors: list[str] = []
    for annotation_dir in candidate_dirs:
        if annotation_dir is None or not annotation_dir.exists():
            errors.append(f"{annotation_dir}: directory not found")
            continue

        formats_to_try = (
            [fmt]
            if fmt in {"doc_grid", "well_gene"}
            else ["doc_grid", "well_gene"]
        )
        for candidate_fmt in formats_to_try:
            try:
                if candidate_fmt == "doc_grid":
                    return _build_doc_grid_annotation(annotation_dir)
                return _build_well_gene_annotation(annotation_dir, plate_name_map)
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{annotation_dir} ({candidate_fmt}): {exc}")

    tried = "\n".join(f"  - {msg}" for msg in errors)
    raise FileNotFoundError(
        "Could not load plate annotations. Tried:\n"
        f"{tried}\n"
        "Expected either towbintools doc/ grids (8x12 ';' CSVs + "
        "rnai_library_genes.csv) or Well,Gene plate CSVs (e.g. in report/)."
    )


def annotate_filemap_with_gene(filemap_df: pl.DataFrame, doc_annot: pl.DataFrame):
    """Merge gene information into the filemap using plate/well."""
    annot = doc_annot.with_columns(
        plate=pl.col("plate_name"),
    ).select(
        ["plate", "well96", "lib_code", "gene_name", "is_control"]
    )

    merged = filemap_df.join(annot, on=["plate", "well96"], how="left")
    return merged


def _add_empty_gene_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add nullable gene annotation columns when plate maps are unavailable."""
    out = df
    if "lib_code" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("lib_code"))
    if "gene_name" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Utf8).alias("gene_name"))
    if "is_control" not in out.columns:
        out = out.with_columns(pl.lit(None).cast(pl.Boolean).alias("is_control"))
    return out


def filemap_has_gene_labels(df: pl.DataFrame) -> bool:
    """True if the filemap has at least one non-empty gene_name."""
    if "gene_name" not in df.columns:
        return False
    labeled = df.filter(
        pl.col("gene_name").is_not_null() & (pl.col("gene_name").str.strip_chars() != "")
    )
    return labeled.height > 0


def _parse_require_gene_annotation(value) -> bool | None:
    """Return True (required), False (skip), or None (try experiment doc/report, else skip)."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "auto"}:
            return None
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return bool(value)


def build_annotated_screen_filemap(
    root_dir,
    pattern="*.tif*",
    analysis_col="analysis_adult_body_v2/ch2_seg",
    exp_folder_regex=None,
    experiment_root=None,
    plate_annotation_dir=None,
    plate_annotation_format="auto",
    plate_name_map=None,
    require_gene_annotation=None,
):
    """Build an annotated filemap from a body or pharynx mask root.

    Plate maps are read from ``experiment_root/doc`` or ``experiment_root/report``
    when present. If they are missing and ``require_gene_annotation`` is not True,
    morphology continues with plate/well metadata only (``gene_name`` left empty).
    """
    root_dir = Path(root_dir)
    if experiment_root is None:
        # Legacy layout: <experiment>/analysis_.../<plate>/masks
        experiment_root = root_dir.parent.parent
    else:
        experiment_root = Path(experiment_root)

    df = get_dir_filemap_screen(
        root_dir,
        pattern=pattern,
        path_column=analysis_col,
        exp_folder_regex=exp_folder_regex,
    )

    require = _parse_require_gene_annotation(require_gene_annotation)
    if require is False:
        logging.info(
            "Skipping plate gene annotations (require_gene_annotation=False)."
        )
        return _add_empty_gene_columns(df)

    try:
        plate_annot = build_plate_annotation(
            experiment_root,
            plate_annotation_dir=plate_annotation_dir,
            plate_annotation_format=plate_annotation_format,
            plate_name_map=plate_name_map,
        )
    except FileNotFoundError as exc:
        if require is True:
            raise
        logging.warning(
            "Plate annotations not found under %s (doc/ or report/); "
            "continuing without gene labels. %s",
            experiment_root,
            exc,
        )
        return _add_empty_gene_columns(df)

    df = annotate_filemap_with_gene(df, plate_annot)
    if not filemap_has_gene_labels(df):
        if require is True:
            raise ValueError(
                f"Plate annotations under {experiment_root} did not assign any gene_name values."
            )
        logging.warning(
            "Plate annotations loaded but no gene_name values were assigned; "
            "gene-level tables will be skipped."
        )
    return df

