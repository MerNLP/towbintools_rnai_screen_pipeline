# Shiny dashboard

After the pipeline produces a **`*_combined.csv`**, use the Shiny app for QC, plotting, and hit calling.

## Start the app on the cluster

```bash
cd ~/towbintools_rnai_screen_pipeline/shiny_app
micromamba activate towbintools_rnai_screen
shiny run app.py --host 127.0.0.1 --port 8000
```

## SSH tunnel from your laptop

```bash
ssh -L 8000:127.0.0.1:8000 username@izblisbon.unibe.ch
```

Open <http://127.0.0.1:8000> in your browser.

## Load data

1. Set **Experiment directory** to the folder that contains `raw/` (the same `experiment_dir` as in your config).
2. Load the **combined CSV** (e.g. `ch2_seg_ch1_seg_combined.csv` from your analysis output or `screen_report/`).

If masks were written to `analysis_output_dir` on shared storage, point the app at the **experiment** path for raw images; mask paths inside the CSV usually point to the shared analysis folder.

The app resolves mask paths back to raw TIFFs for per-well image display and optional mask overlays.

## Features (overview)

- Gene-level and control-level plots
- Strain comparison and log2FC vs empty vector
- Click wells/points to view raw images and segmentation masks
- Optional preload of images for faster navigation
- Day filter when the `day` column is present; hidden for screens without developmental day

## Requirements

The Shiny app uses the same **`towbintools_rnai_screen`** environment as the pipeline. See [Installation](Installation.md).
