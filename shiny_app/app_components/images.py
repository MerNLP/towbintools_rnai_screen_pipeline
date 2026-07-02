"""Image preview loading and mask overlay rendering for the Shiny app."""

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import tifffile as tiff
from scipy import ndimage as ndi

_PIPELINE_SCRIPTS = Path(__file__).resolve().parents[2] / "pipeline_scripts"
if str(_PIPELINE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_SCRIPTS))
from screen_utils import find_exp_plate_subfolder_in_parts  # noqa: E402

# Downsample previews for display speed.
PREVIEW_MAX_WIDTH = int(os.environ.get("SHINY_PREVIEW_MAX_WIDTH", "1536"))
JPEG_QUALITY = int(os.environ.get("SHINY_JPEG_QUALITY", "85"))

try:
    import pyvips

    _HAS_PYVIPS = True
except Exception:
    pyvips = None  # type: ignore
    _HAS_PYVIPS = False

_BBOX_COLS = ("bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col")
_object_csv_cache: dict[str, tuple[float, pl.DataFrame]] = {}
_MAX_OBJECT_CSV_CACHE = 128


def image_backend_name() -> str:
    """Return the active preview image backend ('pyvips' or 'tifffile')."""
    return "pyvips" if _HAS_PYVIPS else "tifffile"


def cache_key(
    mask_path: str,
    channel: int,
    overlay_mode: str,
    alpha: float,
    object_csv_path: str | None = None,
    mask_positive_mode: str | None = None,
) -> tuple:
    """Build stable image-cache key for preview state."""
    a = int(round(float(alpha) * 100))
    return (
        mask_path,
        int(channel),
        str(overlay_mode),
        a,
        str(object_csv_path or ""),
        str(mask_positive_mode or ""),
        int(PREVIEW_MAX_WIDTH),
        image_backend_name(),
    )


def mask_path_to_raw(mask_path: str, experiment_dir: str) -> Path | None:
    """Map a mask TIFF path to the matching raw TIFF under experiment_dir/raw/."""
    try:
        if not experiment_dir or not mask_path:
            return None
        exp = Path(experiment_dir)
        p = Path(mask_path)

        if not p.is_absolute():
            cand = exp / p
            if cand.exists():
                p = cand

        try:
            rel_parts = p.relative_to(exp).parts
        except Exception:
            rel_parts = p.parts

        subfolder = find_exp_plate_subfolder_in_parts(rel_parts)
        if subfolder is not None:
            cand_raw = exp / "raw" / str(subfolder) / p.name
            if cand_raw.exists():
                return cand_raw

        if p.exists() and "raw" in p.parts:
            return p
        return None
    except Exception:
        return None


def _percentile_normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Match legacy contrast: 1st–99.8th percentile stretch to uint8."""
    values = arr.astype(np.float64, copy=False)
    lo, hi = np.percentile(values, 1), np.percentile(values, 99.8)
    if hi > lo:
        values = np.clip((values - lo) / (hi - lo), 0, 1)
    else:
        values = np.zeros_like(values)
    return (values * 255).astype(np.uint8)


def _uint8_grayscale_to_data_uri(arr8: np.ndarray) -> str:
    if _HAS_PYVIPS:
        return _uint8_rgb_to_data_uri(np.stack([arr8, arr8, arr8], axis=-1))

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr8, mode="L").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _uint8_rgb_to_data_uri(rgb: np.ndarray) -> str:
    if _HAS_PYVIPS:
        h, w, bands = rgb.shape
        if bands != 3:
            raise ValueError("expected RGB array")
        image = pyvips.Image.new_from_memory(
            rgb.tobytes(), w, h, 3, "uchar"
        )
        buf = image.jpegsave_buffer(Q=JPEG_QUALITY)
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_grayscale_tifffile(path: Path, channel: int) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tf:
        n_pages = len(tf.pages)
        if n_pages > 1:
            pg = min(channel, n_pages - 1)
            arr = tf.pages[pg].asarray()
        else:
            arr = tf.pages[0].asarray()
            if arr.ndim == 3:
                ch = min(channel, arr.shape[2] - 1)
                arr = arr[:, :, ch]
    return np.squeeze(arr)


def _load_mask_tifffile(path: Path) -> np.ndarray:
    with tiff.TiffFile(str(path)) as tfm:
        m = tfm.pages[0].asarray()
        if m.ndim == 3:
            m = m[0] if m.shape[0] <= 4 else m[:, :, 0]
    return np.squeeze(m)


def _vips_resize_to_max_width(image: "pyvips.Image", max_width: int) -> "pyvips.Image":
    if max_width <= 0 or image.width <= max_width:
        return image
    scale = max_width / image.width
    return image.resize(scale, kernel=pyvips.Kernel.LINEAR)


def _vips_open_grayscale(path: Path, channel: int) -> "pyvips.Image":
    path_str = str(path)
    image = pyvips.Image.new_from_file(path_str, access="sequential")
    try:
        n_pages = int(image.get("n-pages"))
    except Exception:
        n_pages = 1

    if n_pages > 1:
        page = min(max(channel, 0), n_pages - 1)
        image = pyvips.Image.new_from_file(path_str, access="sequential", page=page)
    elif image.bands > 1:
        band = min(max(channel, 0), image.bands - 1)
        image = image.extract_band(band)

    if image.bands > 1:
        image = image.colourspace("b-w")
    return image


def _vips_grayscale_uint8(path: Path, channel: int, max_width: int) -> np.ndarray:
    image = _vips_open_grayscale(path, channel)
    image = _vips_resize_to_max_width(image, max_width)
    arr = np.squeeze(image.numpy())
    if arr.ndim != 2:
        arr = arr.astype(np.float64)
    return _percentile_normalize_uint8(arr)


def _vips_mask_full_array(mask_path: Path) -> np.ndarray:
    """Load full mask raster (pyvips decode is typically faster than tifffile)."""
    image = pyvips.Image.new_from_file(str(mask_path), access="sequential")
    return np.squeeze(image.numpy())


def _vips_mask_array(
    mask_path: Path,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    image = pyvips.Image.new_from_file(str(mask_path), access="sequential")
    image = _vips_resize_to_max_width(image, max(target_width, 1))
    if image.height != target_height or image.width != target_width:
        x_scale = target_width / image.width
        y_scale = target_height / image.height
        image = image.resize(x_scale, vscale=y_scale, kernel=pyvips.Kernel.NEAREST)
    return np.squeeze(image.numpy())


def tiff_to_base64_jpeg(path: Path, channel: int = 0) -> str | None:
    """Load a TIFF preview as a base64 JPEG data URI (downsampled when pyvips is available)."""
    try:
        if _HAS_PYVIPS:
            arr8 = _vips_grayscale_uint8(path, channel, PREVIEW_MAX_WIDTH)
            return _uint8_grayscale_to_data_uri(arr8)

        arr = _load_grayscale_tifffile(path, channel)
        arr8 = _percentile_normalize_uint8(arr)
        return _uint8_grayscale_to_data_uri(arr8)
    except Exception:
        return None


def _read_object_csv(path: Path) -> pl.DataFrame | None:
    try:
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        cached = _object_csv_cache.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        odf = pl.read_csv(str(path), infer_schema_length=1000)
        if len(_object_csv_cache) >= _MAX_OBJECT_CSV_CACHE:
            _object_csv_cache.pop(next(iter(_object_csv_cache)))
        _object_csv_cache[key] = (mtime, odf)
        return odf
    except Exception:
        return None


def _object_qc_component_labels(mask_values: np.ndarray) -> np.ndarray:
    """Connected-component IDs; match pipeline object_label (foreground == 1)."""
    m1 = mask_values == 1
    if not np.any(m1):
        m1 = mask_values > 0
    lbl, _ = ndi.label(m1.astype(bool), structure=np.ones((3, 3), dtype=bool))
    return lbl


def _zoom_nearest_bool(comp: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if target_h <= 0 or target_w <= 0:
        return np.zeros((0, 0), dtype=bool)
    src_h, src_w = comp.shape
    if src_h == target_h and src_w == target_w:
        return comp
    zoomed = ndi.zoom(
        comp.astype(np.float32),
        (target_h / max(src_h, 1), target_w / max(src_w, 1)),
        order=0,
    )
    return zoomed >= 0.5


def _object_qc_from_labels(
    raw8: np.ndarray,
    mask_values: np.ndarray,
    odf: pl.DataFrame,
    *,
    mask_positive_mode: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Color objects green/red from pred_class; uses mask == 1 component IDs."""
    _ = mask_positive_mode

    preview_h, preview_w = raw8.shape
    full_h, full_w = mask_values.shape
    if preview_h == full_h and preview_w == full_w:
        scale_y = scale_x = 1.0
    else:
        scale_y = preview_h / max(full_h, 1)
        scale_x = preview_w / max(full_w, 1)

    lbl = _object_qc_component_labels(mask_values)
    rgb = np.stack([raw8, raw8, raw8], axis=-1).astype(np.float32)
    class_overlay = np.zeros_like(rgb)
    active_mask = np.zeros((preview_h, preview_w), dtype=bool)
    good = np.array([0.0, 200.0, 0.0], dtype=np.float32)
    err = np.array([220.0, 0.0, 0.0], dtype=np.float32)
    has_bbox = all(c in odf.columns for c in _BBOX_COLS)

    rows = odf.select(
        [
            c
            for c in ["object_label", "pred_class", *_BBOX_COLS]
            if c in odf.columns
        ]
    ).drop_nulls(subset=["object_label", "pred_class"])

    for rec in rows.iter_rows(named=True):
        try:
            obj_label = int(rec["object_label"])
            pred_class = str(rec["pred_class"]).strip().lower()
        except Exception:
            continue
        if pred_class == "good":
            color = good
        elif pred_class == "error":
            color = err
        else:
            continue

        if has_bbox:
            try:
                y0 = int(rec["bbox_min_row"])
                y1 = int(rec["bbox_max_row"]) + 1
                x0 = int(rec["bbox_min_col"])
                x1 = int(rec["bbox_max_col"]) + 1
            except Exception:
                continue
            y0 = max(0, min(full_h, y0))
            y1 = max(0, min(full_h, y1))
            x0 = max(0, min(full_w, x0))
            x1 = max(0, min(full_w, x1))
            if y1 <= y0 or x1 <= x0:
                continue
            comp_crop = lbl[y0:y1, x0:x1] == obj_label
            if not np.any(comp_crop):
                continue
            py0 = max(0, int(np.floor(y0 * scale_y)))
            py1 = min(preview_h, int(np.ceil(y1 * scale_y)))
            px0 = max(0, int(np.floor(x0 * scale_x)))
            px1 = min(preview_w, int(np.ceil(x1 * scale_x)))
            th, tw = py1 - py0, px1 - px0
            comp_preview = _zoom_nearest_bool(comp_crop, th, tw)
        else:
            comp_full = lbl == obj_label
            if not np.any(comp_full):
                continue
            if scale_y == 1.0 and scale_x == 1.0:
                comp_preview = comp_full
            else:
                comp_preview = _zoom_nearest_bool(comp_full, preview_h, preview_w)

        if not np.any(comp_preview):
            continue
        if has_bbox:
            class_overlay[py0:py1, px0:px1][comp_preview] = color
            active_mask[py0:py1, px0:px1] |= comp_preview
        else:
            class_overlay[comp_preview] = color
            active_mask |= comp_preview

    if not np.any(active_mask):
        return None, None
    return class_overlay, active_mask


def _overlay_rgb_array(
    raw8: np.ndarray,
    mask_values: np.ndarray,
    *,
    alpha: float,
    mode: str,
    object_csv_path: Path | None,
    mask_positive_mode: str,
    odf: pl.DataFrame | None = None,
) -> np.ndarray | None:
    if mask_positive_mode == "eq1":
        mask = mask_values == 1
    else:
        mask = mask_values > 0

    preview_h, preview_w = raw8.shape
    if mask.shape == (preview_h, preview_w):
        mask_for_simple = mask
    else:
        mask_for_simple = _zoom_nearest_bool(mask, preview_h, preview_w)

    a = float(alpha)
    a = 0.0 if a < 0 else (1.0 if a > 1 else a)

    rgb = np.stack([raw8, raw8, raw8], axis=-1).astype(np.float32)
    class_overlay = None
    active_mask = mask_for_simple

    if object_csv_path is not None and object_csv_path.exists():
        try:
            odf = odf if odf is not None else _read_object_csv(object_csv_path)
            if odf is not None and {"object_label", "pred_class"}.issubset(set(odf.columns)):
                qc_overlay, qc_mask = _object_qc_from_labels(
                    raw8,
                    mask_values,
                    odf,
                    mask_positive_mode=mask_positive_mode,
                )
                if qc_overlay is not None and qc_mask is not None:
                    class_overlay = qc_overlay
                    active_mask = qc_mask
        except Exception:
            class_overlay = None

    if str(mode) == "outline":
        m0 = active_mask
        up = np.zeros_like(m0)
        up[1:] = m0[:-1]
        dn = np.zeros_like(m0)
        dn[:-1] = m0[1:]
        lf = np.zeros_like(m0)
        lf[:, 1:] = m0[:, :-1]
        rt = np.zeros_like(m0)
        rt[:, :-1] = m0[:, 1:]
        er = m0 & up & dn & lf & rt
        draw_mask = m0 & (~er)
    else:
        draw_mask = active_mask

    if class_overlay is not None:
        rgb[draw_mask] = (1 - a) * rgb[draw_mask] + a * class_overlay[draw_mask]
    else:
        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        rgb[draw_mask] = (1 - a) * rgb[draw_mask] + a * red

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _overlay_from_arrays(
    raw8: np.ndarray,
    mask_values: np.ndarray,
    *,
    alpha: float,
    mode: str,
    object_csv_path: Path | None,
    mask_positive_mode: str,
    odf: pl.DataFrame | None = None,
) -> str | None:
    out = _overlay_rgb_array(
        raw8,
        mask_values,
        alpha=alpha,
        mode=mode,
        object_csv_path=object_csv_path,
        mask_positive_mode=mask_positive_mode,
        odf=odf,
    )
    return _uint8_rgb_to_data_uri(out) if out is not None else None


def overlay_mask_on_raw_to_base64_jpeg(
    raw_path: Path,
    mask_path: Path,
    channel: int = 0,
    alpha: float = 0.35,
    mode: str = "fill",
    object_csv_path: Path | None = None,
    mask_positive_mode: str = "nonzero",
) -> str | None:
    """Render raw grayscale with segmentation overlay; color by pred_class when available."""
    try:
        needs_object_colors = (
            object_csv_path is not None and Path(object_csv_path).exists()
        )
        obj_csv_p = Path(object_csv_path) if needs_object_colors else None
        odf = _read_object_csv(obj_csv_p) if obj_csv_p is not None else None

        if needs_object_colors:
            if _HAS_PYVIPS:
                raw8 = _vips_grayscale_uint8(raw_path, channel, PREVIEW_MAX_WIDTH)
                m_full = _vips_mask_full_array(mask_path)
                if m_full.ndim == 3:
                    m_full = m_full[0] if m_full.shape[0] <= 4 else m_full[:, :, 0]
            else:
                raw = _load_grayscale_tifffile(raw_path, channel)
                raw8 = _percentile_normalize_uint8(raw)
                m_full = _load_mask_tifffile(mask_path)
                if raw8.shape[1] > PREVIEW_MAX_WIDTH:
                    preview_scale = PREVIEW_MAX_WIDTH / raw8.shape[1]
                    preview_h = max(1, int(round(raw8.shape[0] * preview_scale)))
                    raw8 = ndi.zoom(
                        raw8.astype(np.float32),
                        (preview_h / raw8.shape[0], preview_scale),
                        order=1,
                    ).astype(np.uint8)

            rgb = _overlay_rgb_array(
                raw8,
                m_full,
                alpha=alpha,
                mode=mode,
                object_csv_path=obj_csv_p,
                mask_positive_mode=mask_positive_mode,
                odf=odf,
            )
            return _uint8_rgb_to_data_uri(rgb) if rgb is not None else None

        if _HAS_PYVIPS:
            raw8 = _vips_grayscale_uint8(raw_path, channel, PREVIEW_MAX_WIDTH)
            mask_values = _vips_mask_array(mask_path, raw8.shape[0], raw8.shape[1])
            return _overlay_from_arrays(
                raw8,
                mask_values,
                alpha=alpha,
                mode=mode,
                object_csv_path=None,
                mask_positive_mode=mask_positive_mode,
            )

        raw = _load_grayscale_tifffile(raw_path, channel)
        raw8 = _percentile_normalize_uint8(raw)
        m = _load_mask_tifffile(mask_path)
        return _overlay_from_arrays(
            raw8,
            m,
            alpha=alpha,
            mode=mode,
            object_csv_path=None,
            mask_positive_mode=mask_positive_mode,
        )
    except Exception:
        return None
