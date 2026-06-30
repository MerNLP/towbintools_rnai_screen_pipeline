# Running your first screen pipeline

## Experiment layout

Your experiment folder should look like:

```text
<experiment_dir>/
  raw/<plate_subdir>/     # one subfolder per plate; TIFF stacks per well
  doc/   and/or   report/ # optional plate maps for gene labels
```

### Plate annotations (optional)

Gene labels are loaded automatically from **`doc/`** or **`report/`** when present:

- **`doc/`** (`doc_grid`): 8×12 `;`-separated CSVs + `rnai_library_genes.csv`
- **`report/`** (`well_gene`): CSV with `Well` and `Gene` columns, one file per plate

```yaml
plate_annotation_dir: "report"
plate_annotation_format: "well_gene"
plate_name_map:
  "20251209_wBT160_AG1": "AG1"
```

Joined on **`plate` + `well96`**. Without maps, well-level and combined CSVs still run; **`gene_level.csv`** is omitted unless you set `require_gene_annotation: true`.

## Configuration

Start from `configs/config_rnai_screen.yaml` in the repository. Copy and edit for your experiment.

### Required path

| Option | Description |
|--------|-------------|
| **experiment_dir** | Top-level experiment folder (must contain `raw/`) |

### Where outputs go

By default, everything is written under **`{experiment_dir}/analysis/`**:

| Output | Default location |
|--------|------------------|
| Body masks | `analysis/ch2_seg/<plate>/` |
| Pharynx masks | `analysis/ch1_seg/<plate>/` |
| Screen CSVs | `analysis/screen_report/` |
| Config backup | `analysis/report/pipeline_backup/` |

To write masks and CSVs elsewhere, set **`analysis_output_dir`**:

```yaml
experiment_dir: "/mnt/towbin.data/shared/agraf/20251216_Kinetix_4x_RNAi_repeat_4/"
analysis_output_dir: "/mnt/towbin.data/shared/mlawrence/rnai_screen/morphology_computation/agraf/20251216"
```

`screen_body_root` and `screen_pharynx_root` still use paths like `analysis/ch2_seg`; they are resolved under `analysis_output_dir`, not under `experiment_dir`.

Override individual CSV paths if needed:

```yaml
screen_filemap_output: ["/path/to/screen_filemap.csv"]
well_output: ["/path/to/ch2_seg_ch1_seg_well_level.csv"]
gene_output: ["/path/to/ch2_seg_ch1_seg_gene_level.csv"]
combined_output: ["/path/to/ch2_seg_ch1_seg_combined.csv"]
```

### Typical workflow block

```yaml
building_blocks:
  - "segmentation"   # body 
  - "segmentation"   # pharynx 
  - "morphology_computation_screen"

rerun_segmentation: [False, False]
rerun_morphology_computation_screen: [True]

pixelsize: [1.625]

segmentation_column: ['raw']
segmentation_method: ["deep_learning", "deep_learning"]
segmentation_channels: [[1], [0]]

model_path: [
  "/path/to/body.ckpt",
  "/path/to/pharynx.ckpt",
]
batch_size: [2, 2]

screen_body_root: ["analysis/ch2_seg"]
screen_pharynx_root: ["analysis/ch1_seg"]

min_body_area_px: [15000]
min_pharynx_area_px: [500]
```

- **`building_blocks`**: order matters (two segmentations, then morphology).
- **`segmentation_channels`**: `[[1], [0]]` = body from channel index 1, pharynx from index 0 (Python 0-based).
- **`pixelsize`**: µm per pixel. Orca / Kinetix 4x: `1.625` (6.5 µm / 4). Nikon Ti2 10x: often `0.65` (check calibration).
- **`batch_size`**: use `2` (or `1`) if some raw TIFFs are corrupt.

### Rerun flags

| Setting | Typical use |
|---------|-------------|
| `rerun_segmentation: [False, False]` | Skip wells that already have mask files |
| `rerun_segmentation: [True, True]` | Re-segment all plates |
| `rerun_morphology_computation_screen: [True]` | Recompute screen CSVs |
| `rerun_morphology_computation_screen: [False]` | Skip morphology if outputs exist |

One list entry per segmentation block for `rerun_segmentation`.

### Plate folder regex (`exp_folder_regex`)

The pipeline parses each subfolder name under `raw/` to get **strain**, **plate**, and optionally **day**. Only the named group **`day`** is stored as a developmental timepoint; a group named **`date`** (acquisition YYYYMMDD) is **not** used as `day`.

If no regex matches, that folder is skipped.

If plate maps are missing, morphology still runs; gene-level CSV is skipped and `gene_name` stays empty.

### Slurm resources

```yaml
sbatch_memory: 128G
sbatch_time: 0-12:00:00
sbatch_cpus: 16
sbatch_gpus: "rtx2080ti:1"
sbatch_nodelist: "izbdelhi"   # optional
```

GPU is requested only for segmentation jobs. Adjust accordingly (`rtx4090`, `rtx6000`, etc.).


## Running the pipeline

```bash
cd ~/towbintools_rnai_screen_pipeline
micromamba activate towbintools_rnai_screen
bash run_pipeline.sh -c configs/config_rnai_screen.yaml
```

Custom config:

```bash
bash run_pipeline.sh -c /path/to/my_config.yaml
```

The config file used for each run is copied to `analysis/report/pipeline_backup/` under the analysis output directory.

## Monitoring jobs

```bash
squeue -u $USER
```

Logs:

- `sbatch_output/pipeline-<jobid>.out` (dispatcher)
- `temp_files/pipeline_<jobid>/sbatch_output/` (per building block)

## After morphology

Screen tables are described in [Screen morphology computation](MorphologyComputationScreen.md). Load the combined CSV in the [Shiny dashboard](ShinyDashboard.md).
