# Screen morphology computation

The **`morphology_computation_screen`** building block pairs body and pharynx masks per field of view, computes morphology measurements, merges plate-map gene annotations when available, and writes screen-level CSV tables.

## Inputs

| Option | Description |
|--------|-------------|
| **screen_body_root** | Body mask root (default `analysis/ch2_seg`, resolved under analysis output dir) |
| **screen_pharynx_root** | Pharynx mask root (default `analysis/ch1_seg`) |
| **exp_folder_regex** | Regex(es) to parse plate folder names under `raw/` |
| **pixelsize** | µm per pixel for area scaling |
| **min_body_area_px** / **max_body_area_px** | Filter body objects by area (pixels) |
| **min_pharynx_area_px** / **max_pharynx_area_px** | Filter pharynx objects by area (pixels) |
| **body_raw_channel** / **pharynx_raw_channel** | Raw channels for optional QC overlays (defaults 1 and 0) |
| **rerun_morphology_computation_screen** | `True` recomputes CSVs; `False` skips if outputs exist |

Plate annotations are loaded automatically from `doc/` or `report/` when present (see [Running your first screen](RunningFirstPipeline.md)).

Optional:

| Option | Description |
|--------|-------------|
| **plate_annotation_dir** | Fixed folder for plate CSVs |
| **plate_annotation_format** | `auto`, `doc_grid`, or `well_gene` |
| **plate_name_map** | Map folder names to annotation plate names |
| **require_gene_annotation** | `true` = fail without maps; `false` = skip genes; default = try maps, continue without |
| **pattern** | Glob for mask files (default `*.tif*`) |

## Outputs

Default location: **`{analysis_subdir}/screen_report/`**

| File | Content |
|------|---------|
| `screen_filemap.csv` | Mask paths, plate, well, strain, optional day, gene columns |
| `ch2_seg_ch1_seg_well_level.csv` | One row per well (body + pharynx metrics merged) |
| `ch2_seg_ch1_seg_gene_level.csv` | Aggregated per gene per plate (only if plate maps exist) |
| `ch2_seg_ch1_seg_combined.csv` | Per-image rows with well/gene fields for the Shiny app |

Channel tags in filenames follow `screen_body_root` / `screen_pharynx_root` (e.g. `ch2_seg`, `ch1_seg`).

Well-level columns include body area (px and µm²), pharynx count and area, and annotation fields when present.

Override paths in the config when using `analysis_output_dir` on shared storage.

## Optional QC classifier

```yaml
enable_object_level_measurements: [False]
enable_qc_good_vs_error: [False]
qc_models_dir: ["/path/to/qc_xgb_models"]
```

When enabled, object-level measurements and an XGBoost good-vs-error classifier can flag problematic body–pharynx pairs. Defaults are off for standard screens.

## Example

```yaml
building_blocks:
  - "morphology_computation_screen"

rerun_morphology_computation_screen: [True]

screen_body_root: ["analysis/ch2_seg"]
screen_pharynx_root: ["analysis/ch1_seg"]

exp_folder_regex:
  - '^(?P<date>\d{8})_wBT(?P<strain>\d+)_(?P<plate>[^_]+)$'

pixelsize: [1.625]
min_body_area_px: [15000]
min_pharynx_area_px: [500]
```

## Shiny dashboard

The **combined CSV** is the main input for the [Shiny dashboard](ShinyDashboard.md).
