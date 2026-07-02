"""Building blocks for the RNAi screen pipeline (segmentation + screen morphology)."""

import os
from abc import ABC
from abc import abstractmethod

import numpy as np

from pipeline_scripts.utils import create_linker_command
from pipeline_scripts.utils import get_input_and_output_files
from pipeline_scripts.utils import get_output_name
from pipeline_scripts.utils import pickle_objects
from pipeline_scripts.utils import run_command

OPTIONS_MAP = {
    "segmentation": [
        "rerun_segmentation",
        "segmentation_column",
        "segmentation_name_suffix",
        "segmentation_method",
        "segmentation_channels",
        "pixelsize",
        "gaussian_filter_sigma",
        "model_path",
        "predict_on_tiles",
        "tiler_config",
        "enforce_n_channels",
        "activation_layer",
        "batch_size",
    ],
    "morphology_computation_screen": [
        "rerun_morphology_computation_screen",
        "screen_body_root",
        "screen_pharynx_root",
        "screen_filemap_output",
        "well_output",
        "gene_output",
        "combined_output",
        "body_object_dir",
        "pharynx_object_dir",
        "enable_object_level_measurements",
        "enable_qc_good_vs_error",
        "qc_models_dir",
        "body_raw_channel",
        "pharynx_raw_channel",
        "min_body_area_px",
        "max_body_area_px",
        "min_pharynx_area_px",
        "max_pharynx_area_px",
        "qc_max_pair_distance_px",
        "pixelsize",
        "pattern",
        "analysis_col",
        "exp_folder_regex",
        "plate_annotation_dir",
        "plate_annotation_format",
        "plate_name_map",
        "require_gene_annotation",
    ],
}

DEFAULT_OPTIONS = {
    "segmentation": {
        "rerun_segmentation": [False],
        "segmentation_column": ["raw"],
        "segmentation_name_suffix": [None],
        "gaussian_filter_sigma": [1.0],
        "predict_on_tiles": [False],
        "tiler_config": [None],
        "enforce_n_channels": [None],
        "activation_layer": [None],
        "model_path": [None],
        "batch_size": [1],
    },
    "morphology_computation_screen": {
        "rerun_morphology_computation_screen": [False],
        "screen_filemap_output": [None],
        "well_output": [None],
        "gene_output": [None],
        "combined_output": [None],
        "body_object_dir": [None],
        "pharynx_object_dir": [None],
        "enable_object_level_measurements": [False],
        "enable_qc_good_vs_error": [False],
        "qc_models_dir": [""],
        "body_raw_channel": [1],
        "pharynx_raw_channel": [0],
        "min_body_area_px": [100],
        "max_body_area_px": [1_000_000],
        "min_pharynx_area_px": [100],
        "max_pharynx_area_px": [10_000],
        "qc_max_pair_distance_px": [2000.0],
        "pattern": ["*.tif*"],
        "analysis_col": [None],
        "exp_folder_regex": [None],
        "plate_annotation_dir": [None],
        "plate_annotation_format": ["auto"],
        "plate_name_map": [None],
        "require_gene_annotation": [None],
    },
}

SUPPORTED_BUILDING_BLOCKS = frozenset(OPTIONS_MAP.keys())


class BuildingBlock(ABC):
    def __init__(
        self,
        name,
        options,
        block_config,
        return_type,
        script_path,
        requires_gpu=False,
        requires_filemap=False,
    ):
        self.name = name
        self.options = options
        self.block_config = block_config
        self.return_type = return_type
        self.script_path = script_path
        self.requires_gpu = requires_gpu
        self.requires_filemap = requires_filemap

    def __str__(self):
        return f"{self.name}: {self.block_config}"

    @abstractmethod
    def get_output_name(self, config, subdir):
        pass

    @abstractmethod
    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        pass

    def create_command(
        self,
        micromamba_path,
        input_pickle_path,
        output_pickle_path,
        pickled_block_config,
        config,
        pickled_filemap_path=None,
    ):
        script_path = self.script_path

        if script_path.endswith(".sh"):
            command = f"bash {script_path} -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config}"
        elif script_path.endswith(".py"):
            command = f"{micromamba_path} run -n towbintools python3 {script_path} -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        else:
            raise ValueError(
                f"Script type of {script_path} is not supported. The pipeline only supports bash or python scripts."
            )

        if pickled_filemap_path is not None:
            command += f" -f {pickled_filemap_path}"
        return command

    def run(self, experiment_filemap, config, subdir=None):
        block_config = self.block_config
        micromamba_path = config.get("micromamba_path", "~/.local/bin/micromamba")
        temp_dir = config["temp_dir"]

        if self.requires_filemap:
            pickled_filemap_path = pickle_objects(
                temp_dir,
                {"path": "experiment_filemap", "obj": experiment_filemap},
            )[0]
        else:
            pickled_filemap_path = None

        if self.return_type == "subdir":
            subdir = self.get_output_name(config, subdir)
            input_files, output_files = self.get_input_and_output_files(
                config, experiment_filemap, subdir
            )

            if len(input_files) != 0:
                (
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                ) = pickle_objects(
                    temp_dir,
                    {"path": f"{self.name}_input_files", "obj": input_files},
                    {"path": f"{self.name}_output_files", "obj": output_files},
                    {"path": f"{self.name}_block_config", "obj": block_config},
                )

                command = self.create_command(
                    micromamba_path,
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                    config,
                    pickled_filemap_path=pickled_filemap_path,
                )

                linker_command = create_linker_command(
                    micromamba_path, temp_dir, subdir
                )

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(
                    micromamba_path, temp_dir, subdir
                )
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return subdir

        elif self.return_type == "csv":
            output_file = self.get_output_name(config, subdir)
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            rerun = (self.block_config[f"rerun_{self.name}"]) or (
                os.path.exists(output_file) is False
            )

            if len(input_files) != 0 and rerun:
                input_pickle_path, pickled_block_config = pickle_objects(
                    temp_dir,
                    {"path": "input_files", "obj": input_files},
                    {"path": "block_config", "obj": block_config},
                )

                command = self.create_command(
                    micromamba_path,
                    input_pickle_path,
                    output_file,
                    pickled_block_config,
                    config,
                    pickled_filemap_path=pickled_filemap_path,
                )

                linker_command = create_linker_command(
                    micromamba_path, temp_dir, output_file
                )

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(
                    micromamba_path, temp_dir, output_file
                )
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return output_file


class SegmentationBuildingBlock(BuildingBlock):
    LEARNING_BASED_METHODS = ("deep_learning", "conv_paint")

    def __init__(self, block_config):
        method = block_config["segmentation_method"]
        if method in self.LEARNING_BASED_METHODS:
            requires_gpu = True
            script_path = "./pipeline_scripts/learning_based_segment.py"
        else:
            raise ValueError(
                f"Segmentation method {method!r} is not supported in the RNAi screen "
                "pipeline. Use 'deep_learning' or 'conv_paint'."
            )

        super().__init__(
            "segmentation",
            OPTIONS_MAP["segmentation"],
            block_config,
            "subdir",
            script_path,
            requires_gpu,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["segmentation_column"],
            "seg",
            channels=self.block_config["segmentation_channels"],
            subdir=subdir,
            return_subdir=True,
            add_raw=False,
            custom_suffix=self.block_config["segmentation_name_suffix"],
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        input_files, output_files = get_input_and_output_files(
            experiment_filemap,
            [self.block_config["segmentation_column"]],
            subdir,
            rerun=self.block_config["rerun_segmentation"],
        )

        return input_files, output_files


class MorphologyComputationScreenBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/compute_morphology_screen.py"
        super().__init__(
            "morphology_computation_screen",
            OPTIONS_MAP["morphology_computation_screen"],
            block_config,
            "csv",
            script_path,
        )

    def get_output_name(self, config, subdir):
        custom = self.block_config.get("screen_filemap_output")
        if custom not in (None, ""):
            return custom
        screen_report_dir = os.path.join(config["analysis_subdir"], "screen_report")
        return os.path.join(
            screen_report_dir,
            f"screen_filemap.{config['report_format']}",
        )

    @staticmethod
    def _resolve_screen_mask_root(path_like, config):
        """Resolve mask root: absolute path, under analysis_subdir, or under experiment_dir."""
        p = str(path_like)
        if os.path.isabs(p):
            return p
        analysis_subdir = config.get("analysis_subdir", config["experiment_dir"])
        if p.startswith("analysis/"):
            return os.path.join(analysis_subdir, p.split("/", 1)[1])
        return os.path.join(analysis_subdir, p)

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        exp_dir = config["experiment_dir"]
        report_format = config["report_format"]

        screen_report_dir = os.path.join(config["analysis_subdir"], "screen_report")
        os.makedirs(screen_report_dir, exist_ok=True)

        body_rel = self.block_config["screen_body_root"]
        body_root = self._resolve_screen_mask_root(body_rel, config)

        pharynx_rel = self.block_config["screen_pharynx_root"]
        pharynx_root = self._resolve_screen_mask_root(pharynx_rel, config)

        self.block_config.setdefault("experiment_dir", exp_dir)

        if self.block_config.get("analysis_col") in (None, ""):
            self.block_config["analysis_col"] = str(body_rel)

        from pipeline_scripts.screen_measurements import extract_channel_tag

        body_tag = extract_channel_tag(str(body_rel))
        pharynx_tag = extract_channel_tag(str(pharynx_rel))

        self.block_config.setdefault("pharynx_root", str(pharynx_root))
        self.block_config.setdefault(
            "well_output",
            os.path.join(
                screen_report_dir,
                f"{body_tag}_{pharynx_tag}_well_level.{report_format}",
            ),
        )
        self.block_config.setdefault(
            "gene_output",
            os.path.join(
                screen_report_dir,
                f"{body_tag}_{pharynx_tag}_gene_level.{report_format}",
            ),
        )
        self.block_config.setdefault(
            "combined_output",
            os.path.join(
                screen_report_dir,
                f"{body_tag}_{pharynx_tag}_combined.{report_format}",
            ),
        )

        input_files = [body_root]
        return input_files, None


def count_building_blocks_types(building_block_names):
    building_block_counts = {}
    for i, building_block in enumerate(building_block_names):
        if building_block not in building_block_counts:
            building_block_counts[building_block] = []
        building_block_counts[building_block] += [i]
    return building_block_counts


def parse_building_blocks_config(config):
    building_block_names = config["building_blocks"]
    building_block_counts = count_building_blocks_types(building_block_names)

    blocks_config = {}

    for i, building_block_name in enumerate(building_block_names):
        config_copy = config.copy()
        if building_block_name not in SUPPORTED_BUILDING_BLOCKS:
            raise ValueError(
                f"Unknown building block {building_block_name!r}. "
                f"Supported: {sorted(SUPPORTED_BUILDING_BLOCKS)}"
            )

        options = OPTIONS_MAP[building_block_name]
        for option in options:
            try:
                assert (
                    len(config[option])
                    == len(building_block_counts[building_block_name])
                    or len(config[option]) == 1
                ), (
                    f"{config[option]} The number of {option} options "
                    f"({len(config[option])}) does not match the number of "
                    f"{building_block_name} building blocks "
                    f"({len(building_block_counts[building_block_name])})"
                )

            except KeyError:
                if option in DEFAULT_OPTIONS[building_block_name]:
                    config_copy[option] = DEFAULT_OPTIONS[building_block_name][option]
                    print(
                        f"{option} not found in config file, using default value: "
                        f"{config_copy[option]}"
                    )
                else:
                    raise KeyError(
                        f"{option} is not in the config file, but is required for "
                        f"the {building_block_name} building block."
                    ) from None

        for option in options:
            if len(config_copy[option]) == 1:
                config_copy[option] = config_copy[option] * len(
                    building_block_counts[building_block_name]
                )

        idx = np.argwhere(
            np.array(building_block_counts[building_block_name]) == i
        ).squeeze()

        block_options = {option: config_copy[option][idx] for option in options}
        block_options["name"] = building_block_name

        blocks_config[i] = block_options

    return blocks_config


def create_building_blocks(blocks_config):
    building_blocks = []
    for block_config in blocks_config.values():
        name = block_config["name"]
        if name == "segmentation":
            building_block = SegmentationBuildingBlock(block_config)
        elif name == "morphology_computation_screen":
            building_block = MorphologyComputationScreenBuildingBlock(block_config)
        else:
            raise ValueError(
                f"Building block {name!r} is not supported. "
                f"Supported: {sorted(SUPPORTED_BUILDING_BLOCKS)}"
            )
        building_blocks.append(building_block)

    return building_blocks


def parse_and_create_building_blocks(config):
    blocks_config = parse_building_blocks_config(config)
    return create_building_blocks(blocks_config)
