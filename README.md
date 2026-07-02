# TOWBINTOOLS RNAI SCREEN PIPELINE

Pipeline for **RNAi morphology screens**: body + pharynx deep-learning segmentation, well/gene-level morphology CSVs, and a Shiny dashboard for QC and hit calling.

This is a focused fork of the lab [towbintools_pipeline](https://github.com/spsalmon/towbintools_pipeline). It includes only **segmentation** and **morphology_computation_screen** (no straightening, molt detection, fluorescence, etc.).

The documentation for the package used as a backbone for the pipeline can be found here: <https://towbintools.readthedocs.io/en/latest/towbintools.html>

## Documentation

The pipeline book is here: <https://mernlp.github.io/towbintools_rnai_screen_pipeline/>

Read it before your first run. In particular:

- **Getting started → Installation** — micromamba, conda env `towbintools_rnai_screen`, VS Code on the cluster
- **Getting started → Running your first screen** — experiment layout, YAML config, plate annotations, `exp_folder_regex`, Slurm, rerun flags
- **Building blocks** — segmentation and `morphology_computation_screen`
- **Downstream analysis → Shiny dashboard** — loading combined CSVs, QC, hit calling

For shared concepts (Slurm job chaining, YAML list semantics), see the main lab pipeline book: <https://spsalmon.github.io/towbintools_pipeline/>.

If something is unclear, ask and we will update the book.

## How to install ?

```bash
git clone https://github.com/MerNLP/towbintools_rnai_screen_pipeline.git
cd towbintools_rnai_screen_pipeline
```

Then follow **Installation** in the book (micromamba + `requirements/rnai_screen/install_environment.sh`).

If `install_environment.sh` fails because `conda-lock.yml` is missing, see **Installation** in the book: run `./generate_lock.sh` in `requirements/rnai_screen/`, or create the env from `environment.yml` directly.

## Running the pipeline

1. Read the book (link above).
2. Edit `configs/config_rnai_screen.yaml` for your experiment.
3. Run:

```bash
cd ~/towbintools_rnai_screen_pipeline
micromamba activate towbintools_rnai_screen
bash run_pipeline.sh -c configs/config_rnai_screen.yaml
```

### Using a custom config file

```bash
bash run_pipeline.sh -c path_to_config_file
```

or

```bash
bash run_pipeline.sh --config path_to_config_file
```

## Updating the pipeline

```bash
cd ~/towbintools_rnai_screen_pipeline
git pull
```

Re-run the install script if `requirements/rnai_screen/environment.yml` changed. To upgrade **towbintools**:

```bash
micromamba activate towbintools_rnai_screen
pip install --upgrade towbintools
```
