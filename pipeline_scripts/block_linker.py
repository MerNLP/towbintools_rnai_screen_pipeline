"""Chain RNAi screen pipeline blocks and update the experiment filemap between steps."""

import argparse
import os

import polars as pl
from towbintools.foundation.file_handling import add_dir_to_experiment_filemap
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap

from pipeline_scripts.utils import cleanup_files
from pipeline_scripts.utils import list_image_paths
from pipeline_scripts.utils import load_pickles
from pipeline_scripts.utils import map_paths_by_basename
from pipeline_scripts.utils import pickle_objects
from pipeline_scripts.utils import sync_backup_folder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--temp_dir",
        help="Path to the directory storing temporary files",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--result",
        help="Path to the directory / csv file storing the previous block's results",
        required=False,
    )
    return parser.parse_args()


def cleanup_temp_pickles(pickle_dir, keep_paths):
    """Remove all pickle files in pickle_dir except those in keep_paths."""
    all_pickles = [
        os.path.join(pickle_dir, f)
        for f in os.listdir(pickle_dir)
        if f.endswith(".pkl")
    ]
    pickles_to_delete = [f for f in all_pickles if f not in keep_paths]
    cleanup_files(*pickles_to_delete)


def update_experiment_filemap(
    experiment_filemap: pl.DataFrame,
    config,
    result,
    previous_block,
):
    """Update experiment_filemap based on the previous block's return type."""
    filemap_path = config["filemap_path"]

    if previous_block.return_type == "subdir":
        column_name = (
            f'{config["analysis_dir_name"]}/'
            f"{os.path.basename(os.path.normpath(result))}"
        )
        no_timepoint = config.get("no_timepoint", False) or config.get(
            "no_timepoints", False
        )
        if no_timepoint or ("Time" not in experiment_filemap.columns):
            mask_paths = list_image_paths(result)
            raw_col = config.get("raw_dir_name", "raw")
            if raw_col in experiment_filemap.columns:
                basenames = [
                    os.path.basename(str(path))
                    for path in experiment_filemap[raw_col].to_list()
                ]
                image_paths = map_paths_by_basename(mask_paths, basenames)
            else:
                image_paths = mask_paths
            experiment_filemap = experiment_filemap.with_columns(
                pl.Series(name=column_name, values=image_paths)
            )
        else:
            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap, result, column_name
            )
        write_filemap(experiment_filemap, filemap_path)

    elif previous_block.return_type == "csv":
        if previous_block.name != "morphology_computation_screen":
            raise ValueError(
                f"Unexpected CSV building block {previous_block.name!r}. "
                "Only morphology_computation_screen is supported."
            )
        print(
            "### Skipping analysis_filemap merge for morphology_computation_screen "
            "(screen outputs are written to configured paths) ###"
        )

    else:
        raise ValueError(
            f"Unsupported return type {previous_block.return_type!r} "
            f"for block {previous_block.name!r}."
        )

    return experiment_filemap


def main():
    args = get_args()
    temp_dir = args.temp_dir
    result = args.result

    pickle_dir = os.path.join(temp_dir, "pickles")
    progress_pickle_path = os.path.join(pickle_dir, "progress_tracker.pkl")

    cleanup_temp_pickles(pickle_dir, [progress_pickle_path])

    progress_tracker = load_pickles(progress_pickle_path)[0]

    current_block_index = progress_tracker["current_block_index"]
    building_blocks = progress_tracker["building_blocks"]

    progress_tracker["current_block_index"] += 1
    pickle_objects(temp_dir, {"path": "progress_tracker", "obj": progress_tracker})

    if current_block_index > 0:
        previous = building_blocks[current_block_index - 1]
        previous_block, previous_config = (
            previous["block"],
            previous["config"],
        )

        sync_backup_folder(
            previous_config["temp_dir"], previous_config["pipeline_backup_dir"]
        )

        experiment_filemap = read_filemap(previous_config["filemap_path"])
        update_experiment_filemap(
            experiment_filemap,
            previous_config,
            result,
            previous_block,
        )

    if current_block_index < len(building_blocks):
        current = building_blocks[current_block_index]
        current_building_block, current_subdir, current_config = (
            current["block"],
            current["subdir"],
            current["config"],
        )
        experiment_filemap = read_filemap(current_config["filemap_path"])
        print(f"Running {current_building_block} ...")
        current_building_block.run(
            experiment_filemap, current_config, subdir=current_subdir
        )
    else:
        print("End of the pipeline!")


if __name__ == "__main__":
    main()
