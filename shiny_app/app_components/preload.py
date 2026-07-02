"""Background image preview cache for the Shiny app."""

import threading

from app_components.helpers import _resolve_existing_path
from app_components.images import (
    cache_key,
    mask_path_to_raw,
    overlay_mask_on_raw_to_base64_jpeg,
    tiff_to_base64_jpeg,
)

# Shared across sessions; populated by preload_worker threads.
image_cache: dict[tuple, str] = {}
cache_lock = threading.Lock()
cache_stats: dict[str, int] = {
    "done": 0,
    "total": 0,
    "priority_done": 0,
    "priority_total": 0,
    "priority_complete": 0,
}
preload_threads: list[threading.Thread] = []
N_PRELOAD_THREADS = 4


def _bump_stats(*, is_priority: bool) -> None:
    with cache_lock:
        cache_stats["done"] += 1
        if is_priority:
            cache_stats["priority_done"] += 1
            if (
                cache_stats["priority_total"] > 0
                and cache_stats["priority_done"] >= cache_stats["priority_total"]
            ):
                cache_stats["priority_complete"] = 1


def _cache_preview(
    mask_path: str,
    exp_dir: str,
    channel: int,
    *,
    overlay_mode: str,
    overlay_alpha: float,
    object_csv_path: str | None,
    mask_positive_mode: str,
) -> None:
    """Load one preview variant (raw or overlay) into image_cache if missing."""
    mode = str(overlay_mode or "none")
    alpha = float(overlay_alpha)
    obj_csv_raw = str(object_csv_path or "") or None
    key = cache_key(
        mask_path,
        channel,
        mode,
        alpha,
        obj_csv_raw,
        mask_positive_mode,
    )
    if key in image_cache:
        return

    raw_path = mask_path_to_raw(mask_path, exp_dir)
    if not raw_path or not raw_path.exists():
        return

    data_uri: str | None = None
    if mode == "none":
        data_uri = tiff_to_base64_jpeg(raw_path, channel=channel)
    else:
        mask_p = _resolve_existing_path(mask_path, exp_dir)
        obj_csv_p = (
            _resolve_existing_path(obj_csv_raw, exp_dir) if obj_csv_raw else None
        )
        if mask_p and mask_p.exists():
            data_uri = overlay_mask_on_raw_to_base64_jpeg(
                raw_path,
                mask_p,
                channel=channel,
                alpha=alpha,
                mode=mode,
                object_csv_path=obj_csv_p,
                mask_positive_mode=mask_positive_mode,
            )
        if data_uri is None:
            data_uri = tiff_to_base64_jpeg(raw_path, channel=channel)

    if data_uri:
        with cache_lock:
            image_cache[key] = data_uri


def preload_variants_for_row(
    mask_path: str | None,
    exp_dir: str,
    channel: int,
    *,
    overlay_mode: str,
    overlay_alpha: float,
    object_csv_path: str | None,
    mask_positive_mode: str,
) -> list[str]:
    """Return cache variant names still missing for this mask path."""
    if not mask_path:
        return []
    missing: list[str] = []
    if cache_key(mask_path, channel, "none", 0.0) not in image_cache:
        missing.append("raw")
    mode = str(overlay_mode or "none")
    if mode != "none":
        obj_csv_raw = str(object_csv_path or "") or None
        key = cache_key(
            mask_path,
            channel,
            mode,
            float(overlay_alpha),
            obj_csv_raw,
            mask_positive_mode,
        )
        if key not in image_cache:
            missing.append("overlay")
    return missing


def preload_worker(
    file_rows: list[dict],
    exp_dir: str,
    channel: int,
    *,
    overlay_mode: str = "none",
    overlay_alpha: float = 0.35,
    mask_positive_mode: str = "eq1",
) -> None:
    """Background thread: cache raw previews and optional mask overlays."""
    mode = str(overlay_mode or "none")
    alpha = float(overlay_alpha)
    mask_mode = str(mask_positive_mode or "eq1")

    for r in file_rows:
        mask_path = r.get("_fcol_")
        is_priority = int(r.get("__prio", 2)) <= 1
        obj_csv = r.get("_obj_csv")
        obj_csv_str = str(obj_csv) if obj_csv not in (None, "") else None

        variants = preload_variants_for_row(
            mask_path,
            exp_dir,
            channel,
            overlay_mode=mode,
            overlay_alpha=alpha,
            object_csv_path=obj_csv_str,
            mask_positive_mode=mask_mode,
        )
        if not variants:
            continue

        if not mask_path:
            for _ in variants:
                _bump_stats(is_priority=is_priority)
            continue

        if "raw" in variants:
            _cache_preview(
                mask_path,
                exp_dir,
                channel,
                overlay_mode="none",
                overlay_alpha=0.0,
                object_csv_path=None,
                mask_positive_mode=mask_mode,
            )
            _bump_stats(is_priority=is_priority)
        if "overlay" in variants:
            _cache_preview(
                mask_path,
                exp_dir,
                channel,
                overlay_mode=mode,
                overlay_alpha=alpha,
                object_csv_path=obj_csv_str,
                mask_positive_mode=mask_mode,
            )
            _bump_stats(is_priority=is_priority)
