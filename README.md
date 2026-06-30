# TOWBINTOOLS RNAI SCREEN PIPELINE

Pipeline for **RNAi morphology screens**: body + pharynx deep-learning segmentation, screen morphology CSVs, and a Shiny dashboard for QC and hit calling.

It is a **focused fork** of the lab [towbintools_pipeline](https://github.com/spsalmon/towbintools_pipeline) — only segmentation and `morphology_computation_screen` are included (no straightening, molt detection, fluorescence, etc.).

The underlying Python package is documented here: <https://towbintools.readthedocs.io/en/latest/towbintools.html>

## RTFM

Documentation for this pipeline is built with **MyST** (Jupyter Book) from the `book/` folder:

**<https://mernlp.github.io/towbintools_rnai_screen_pipeline/>**

*(Live after you publish the repo and enable GitHub Pages — see `.github/workflows/deploy_book.yml`.)*

To build locally:

```bash
npm install -g jupyter-book
cd book && jupyter-book build --html
```

Open `book/_build/html/index.html` in a browser.

For **general pipeline concepts** (Slurm job chaining, YAML list semantics), see the main lab pipeline book: <https://spsalmon.github.io/towbintools_pipeline/>.

If you don't understand something, feel free to ask, and we'll update the wiki to make it clearer !

## How to install ?

### How to set up Visual Studio Code ?

**Essentially the same steps as for `towbintools_pipeline`.**

1. Download VS Code: <https://code.visualstudio.com/download>
2. Install it like you would install any software.
3. Inside of VS Code, open a terminal and run :

```bash
code --install-extension ms-vscode-remote.remote-ssh
```

Now, click on the remote explorer icon that should be on the left of the window and click on the + to add a new remote.
Enter the command you usually use to ssh into the cluster using PuTTY, for example:

```bash
ssh username@izblisbon.unibe.ch
```

Change username to your username (first letter of your first name + last name, eg : mlawrence)

Optionnal, but **HIGHLY** recommended. Open the Windows command line (cmd). Run :

```bash
ssh-keygen
```

- Select all the default options, except if you are extremely paranoid and want to set a passphrase.
  Go into the folder where the file was saved, it should be something like Users/username/.ssh/

- Open the file **id_rsa.pub** using the notepad or any text editing software.
  Copy the entire content of the file.

- In VS Code, go to your home folder : /home/username/

- Go into the .ssh folder

- If it doesn't exist, create a file named **authorized_keys**

- Paste the content of the **id_rsa.pub** file that you copied earlier into this file

- You will now be able to connect to the cluster without having to type your password

If you want to code using Python, you should run the following commands, while connected inside of VS Code, while being connected to your session on the cluster.

```bash
code --install-extension ms-python.python
```

```bash
code --install-extension ms-toolsai.jupyter
```

```bash
code --install-extension ms-python.vscode-pylance
```

### How to install the pipeline itself

**Same overall flow as `towbintools_pipeline`** (clone → micromamba → install script), but this repo and env have different names.

- In VS Code, open a terminal and cd to your home directory :

```bash
cd
```

- Clone this repo from GitHub :

```bash
git clone https://github.com/MerNLP/towbintools_rnai_screen_pipeline.git
```

- Install micromamba and restart your shell (skip if you already have it from `towbintools_pipeline`) :

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

```bash
source ~/.bashrc
```

- Run the installation script :

```bash
chmod +x ~/towbintools_rnai_screen_pipeline/requirements/install_environment.sh
```

```bash
cd ~/towbintools_rnai_screen_pipeline/requirements
```

```bash
./install_environment.sh
```

This creates the conda environment **`towbintools_rnai_screen`**.

If the lock file is missing, generate it first (on a machine with network access):

```bash
./generate_lock.sh
```

Or install directly from `environment.yml`:

```bash
micromamba create -n towbintools_rnai_screen -f environment.yml -y
```

For some reason, the script doesn't really work for some people. In case it doesn't work for you, just run every line of the installation script manually.

Follow the directions given, so basically, push enter a bunch of times and type yes (you want to answer yes everytime) when asked to.

**Note:** Segmentation needs **towbintools >= 0.4.1** and GPU nodes. Model checkpoints are **not** in this repo — set `model_path` in your config to the shared `.ckpt` files on the cluster.

## Experiment layout

Your experiment folder should contain at least:

```text
<experiment_dir>/
  raw/<plate_subdir>/   # TIFF stacks per plate
  doc/   and/or   report/   # plate maps (see RTFM above)
```

## Running the pipeline

**Same command pattern as `towbintools_pipeline`.**

1. Read the WIKI !!!!!
2. Modify `configs/config_rnai_screen.yaml` according to what you want to do.
3. Run:

```bash
cd ~/towbintools_rnai_screen_pipeline
```

```bash
micromamba activate towbintools_rnai_screen
```

```bash
bash run_pipeline.sh -c configs/config_rnai_screen.yaml
```

### Using a custom config file

If you don't specify anything, the default config is `./configs/config.yaml` but the usual screen config is `configs/config_rnai_screen.yaml`. You can specify any config using

```bash
bash run_pipeline.sh -c path_to_config_file
```

or

```bash
bash run_pipeline.sh --config path_to_config_file
```

### Resume / rerun flags (screen)

| Setting | Typical use |
|---------|-------------|
| `rerun_segmentation: [False, False]` | Skip plates/wells that already have masks |
| `rerun_segmentation: [True, True]` | Re-segment everything |
| `rerun_morphology_computation_screen: [True]` | Recompute screen CSVs |
| `experiment_subdirs: [...]` | Limit to specific plates (smoke tests) |

### Monitor jobs

**Same as `towbintools_pipeline`:**

```bash
squeue -u $USER
```

Logs: `sbatch_output/pipeline-<jobid>.out` and `temp_files/pipeline_<jobid>/sbatch_output/`.

## Documentation

| Book | What it covers |
|------|----------------|
| [towbintools_rnai_screen_pipeline book](https://mernlp.github.io/towbintools_rnai_screen_pipeline/) | Install, config, plate maps, morphology CSVs, Shiny app |
| [towbintools_pipeline book](https://spsalmon.github.io/towbintools_pipeline/) | Shared pipeline concepts — building blocks, Slurm, segmentation |

Source: `book/` (MyST). Deployed via GitHub Actions on push to `main`.

## Running the Shiny dashboard

After morphology has produced a `*_combined.csv`:

```bash
cd ~/towbintools_rnai_screen_pipeline/shiny_app
micromamba activate towbintools_rnai_screen
shiny run app.py --host 127.0.0.1 --port 8000
```

From your laptop, tunnel to the cluster (same idea as any remote web app):

```bash
ssh -L 8000:127.0.0.1:8000 username@izblisbon.unibe.ch
```

Then open <http://127.0.0.1:8000>, set **Experiment directory** and load the **combined CSV**.

## Updating the pipeline

If you use git to pull updates to this repo, **same idea as `towbintools_pipeline`**: pull latest code, then refresh the environment if `requirements/environment.yml` changed.

```bash
cd ~/towbintools_rnai_screen_pipeline
git pull
```

Re-run `requirements/install_environment.sh` or update packages manually if needed.

### Updating the towbintools package

First activate the environment :

```bash
micromamba activate towbintools_rnai_screen
```

Then upgrade the package (use a version compatible with this pipeline, e.g. 0.4.1+):

```bash
pip install --upgrade towbintools
```

## Known limits

- **Corrupt raw TIFFs**: segmentation may write black placeholder masks and continue (`batch_size: 2` helps; see `learning_based_segment.py` patches). Replace bad files at source when possible.
- **Slurm settings** (`sbatch_cpus`, `sbatch_memory`, `sbatch_gpus`, `sbatch_nodelist`) are cluster-specific — adjust in your config.
