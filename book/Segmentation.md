# Segmentation

Segmentation extracts body and pharynx masks from raw well images. Each segmentation building block runs one model on one channel.

The screen pipeline supports **`deep_learning`** segmentation only (no edge-based or threshold methods).

## Options

| Option | Description |
|--------|-------------|
| **segmentation_column** | Filemap column with image paths (usually `"raw"`) |
| **segmentation_method** | `"deep_learning"` |
| **segmentation_channels** | Channel index(es) to segment, 0-based. `[[1], [0]]` = body from 2nd channel, pharynx from 1st |
| **segmentation_name_suffix** | Optional suffix on output folder name. Default: `null` |
| **model_path** | Path to Lightning `.ckpt` checkpoint |
| **batch_size** | Images per GPU forward pass. Increase for speed; decrease if you run out of VRAM |
| **rerun_segmentation** | `False` skips wells that already have mask files; `True` re-segments everything |

Advanced options (`predict_on_tiles`, `tiler_config`, `enforce_n_channels`, `activation_layer`) are passed through to towbintools when needed. See the main pipeline segmentation page: <https://spsalmon.github.io/towbintools_pipeline/segmentation/>.

## Default output folders

For `segmentation_channels: [[1], [0]]` with `segmentation_column: ['raw']`:

```text
{experiment_dir}/analysis/ch2_seg/<plate_subdir>/   # body
{experiment_dir}/analysis/ch1_seg/<plate_subdir>/   # pharynx
```

These paths are referenced by `screen_body_root` and `screen_pharynx_root` in the morphology block.

## Example

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"

segmentation_column: ['raw']
segmentation_method: ["deep_learning", "deep_learning"]
segmentation_channels: [[1], [0]]
segmentation_name_suffix: [null, null]

model_path: [
  "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/body/towbintools_medium/best_light.ckpt",
  "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/pharynx/towbintools_medium/best_light.ckpt",
]
batch_size: [2, 2]
rerun_segmentation: [False, False]
```

## Corrupt TIFFs

If a raw TIFF cannot be read, towbintools may skip it or fail the whole batch depending on version and `batch_size`. This pipeline patches `learning_based_segment.py` to tolerate partial batch failure.

Practical tips:

- Set **`batch_size: [2, 2]`** (or `1`) when some wells have bad files
- Replace corrupt TIFFs at source when possible
- Segmentation may write black placeholder masks for failed images so morphology can continue

Requires **towbintools >= 0.4.1** and a GPU Slurm job.
