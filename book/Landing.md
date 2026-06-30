# towbintools_rnai_screen_pipeline

A focused pipeline for **RNAi morphology screens** in *C. elegans*: deep-learning segmentation of body and pharynx, well- and gene-level morphology tables, and a Shiny dashboard for QC and hit calling.

This repository is a slim fork of the lab [towbintools_pipeline](https://github.com/spsalmon/towbintools_pipeline). It keeps only:

- **segmentation** (body + pharynx)
- **morphology_computation_screen**

There is no straightening, molt detection, fluorescence quantification, or per-worm timelapse morphology.

Head to **Getting started** for [installation](Installation.md) and [running your first screen](RunningFirstPipeline.md) (config, plate annotations, regex, Slurm).

For shared concepts (Slurm job chaining, YAML list semantics, troubleshooting), see the main pipeline documentation: <https://spsalmon.github.io/towbintools_pipeline/>.

The underlying Python package API is documented at <https://towbintools.readthedocs.io/en/latest/towbintools.html>.
