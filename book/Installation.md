# Installing the pipeline

## 1. Clone the repository

```bash
git clone https://github.com/MerNLP/towbintools_rnai_screen_pipeline.git
cd ~/towbintools_rnai_screen_pipeline
```

## 2. Install micromamba

Skip this step if you already use micromamba (for example from `towbintools_pipeline`).

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

## 3. Create the conda environment

```bash
cd ~/towbintools_rnai_screen_pipeline/requirements/rnai_screen
chmod +x install_environment.sh
./install_environment.sh
```

This creates the environment **`towbintools_rnai_screen`**.

If `conda-lock.yml` is missing, generate it first (on a machine with network access):

```bash
./generate_lock.sh
```

Or install directly from `environment.yml`:

```bash
micromamba create -n towbintools_rnai_screen -f environment.yml -y
```

Activate the environment:

```bash
micromamba activate towbintools_rnai_screen
```

## 4. VS Code on the cluster (optional)

Use the [Remote SSH](https://code.visualstudio.com/docs/remote/ssh) extension to connect to `username@izblisbon.unibe.ch`. The workflow matches the main [towbintools_pipeline](https://github.com/spsalmon/towbintools_pipeline) install guide.

Recommended extensions: Python, Jupyter, Pylance.

## 5. Model checkpoints

Segmentation models are **not** shipped with this repository. Point `model_path` in your config to shared `.ckpt` files on the cluster (see [Segmentation](Segmentation.md)).

Example (paper models on shared storage):

```text
/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/body/towbintools_medium/best_light.ckpt
/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/pharynx/towbintools_medium/best_light.ckpt
```

Segmentation requires **towbintools >= 0.4.1** and a GPU node.

## Next step

[Run your first screen pipeline](RunningFirstPipeline.md).
