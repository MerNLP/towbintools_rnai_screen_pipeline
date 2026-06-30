# Building blocks

A building block is one atomic step in the pipeline (segment this channel, compute screen morphology, etc.).

The RNAi screen pipeline supports three block types:

- [segmentation](Segmentation.md) — deep-learning masks for body and pharynx
- **morphology_computation_screen** — well-, gene-, and combined-level screen tables

For how building blocks chain on Slurm and how the dispatcher submits jobs, see the main pipeline book: <https://spsalmon.github.io/towbintools_pipeline/buildingblock/>.

## Configuration lists

All block options are **lists** in the YAML config.

- A list with **one element** is broadcast to every block of that type.
- A list with **one element per block** assigns options in order.

Example: two segmentations sharing one method:

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"
segmentation_method: ["deep_learning"]
```

Example: body and pharynx with different channels and models:

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"
  - "morphology_computation_screen"

segmentation_method: ["deep_learning", "deep_learning"]
segmentation_channels: [[1], [0]]
model_path: ["body.ckpt", "pharynx.ckpt"]
batch_size: [2, 2]
rerun_segmentation: [False, False]
rerun_morphology_computation_screen: [True]
```

Use **`null`** for an empty optional value.

Morphology-only options (`screen_body_root`, `exp_folder_regex`, `min_body_area_px`, etc.) use a **single list entry** because there is only one `morphology_computation_screen` block.

## Output naming

Segmentation writes mask TIFFs under `analysis_subdir` (default `{experiment_dir}/analysis/`, or `analysis_output_dir` if set):

| Channel index | Folder |
|---------------|--------|
| 0 | `ch1_seg/<plate>/` |
| 1 | `ch2_seg/<plate>/` |

The morphology block reads masks via `screen_body_root` / `screen_pharynx_root` (e.g. `analysis/ch2_seg`) and writes CSVs under `analysis/screen_report/` unless paths are overridden.

Details: [Segmentation](Segmentation.md), [Screen morphology](MorphologyComputationScreen.md).
