import pathlib
import logging
from os import environ

from box import Box


logger = logging.getLogger(__name__)


def _merge_boxes(left: Box, right: Box) -> Box:
    for k, v in right.items():
        if k not in left:
            left[k] = v
        else:
            if isinstance(v, Box):
                left[k] = _merge_boxes(left[k], v)
            else:
                left[k] = v
    return left


class ConfigReader:
    name: str | None
    env_vars: dict[str, str]

    def __init__(self, name: str | None):
        """Config Reader Class

        Each reader is created with a name that determines how it loads environment
        variables.
        If no name is provided then no environment variables are loaded.
        """
        self.name = name

    def load(self, base_path: str | pathlib.Path) -> Box:
        """Load a config structure

        It performs three processing steps:
            - Loading and replacing environment variables
            - Processing paths and converting to pathlib objects
            - Converting ranges to lists

        It loads all environment variables fitting "$name_a__b_c", where "$name_"
        is removed and "__" is replaced with dots to allow for deeper insertions
        into the box.
        Since environment variables are case insensitive, the string is cast to lower case.

        The loader can insert paths into other paths or file names.
        This is performed first for the `data_base_path`, then for the paths in `path`.
        At this time only the data base path can be used for interpolation.
        Finally, it is done for all other paths where both the data base path and
        the paths in `paths` can be used.
        Interpolation can be performed as `"{data_base_path}/result.csv"`.
        Note, that the quotations are important.

        Ranges can be specified as a dict with two entries `start` and `end`.
        This dict is replaced with a list, where the range is included.

        Args:
            base_path: The path where the base config is stored.
        """

        if isinstance(base_path, str):
            base_path = pathlib.Path(base_path)
        if not base_path.exists():
            raise ValueError(f"Path {base_path} does not exist")
        if not (base_path / "config.yaml").exists():
            raise ValueError(f"Path {base_path} does not contain config.yaml")
        # Load all config files in this directory.
        config = Box.from_yaml(filename=base_path / "config.yaml")
        for file in base_path.glob("*.yaml"):
            if file.name == "config.yaml":
                continue
            subname = file.stem
            if subname in config:
                raise ValueError(f"Subconfig '{subname}' already exists in config!")
            config[subname] = Box.from_yaml(filename=file)
        # Process the config
        if self.name is not None:
            self._parse_environ_overwrites(config)
        self._parse_paths(config)
        self._convert_range(config)
        return config

    def _parse_environ_overwrites(self, config: Box):
        prefix = f"{self.name.upper()}_"
        for k, val in environ.items():
            if k.startswith(prefix):
                name = k.replace(prefix, "").replace("__", ".").lower()
                logger.info(f"Setting {name} to '{val}'.")
                config[name] = val
        return

    def _parse_paths(self, config: Box):
        """Process paths: Interpolate, Convert and check (only paths)."""
        path_dict = {}
        if "data_base_path" in config:
            data_pase_path = pathlib.Path(config["data_base_path"])
            if not data_pase_path.exists():
                raise ValueError(f"Data base path '{data_pase_path}' does not exist")
            path_dict["data_base_path"] = str(data_pase_path)
            config["data_base_path"] = data_pase_path
        if "paths" in config:
            for name, path in config["paths"].items():
                path = pathlib.Path(path.format(**path_dict))
                if not path.exists():
                    logger.warning(f"Path '{path}' does not exist.")
                path_dict[name] = str(path)
                config.paths[name] = path
        # Go through everything else.
        self._convert_paths_inner(config, path_dict)
        return

    def _convert_paths_inner(self, config: Box, path_dict: dict[str, str]):
        """Recursively convert all paths and filenames to pathlib objects."""
        for k, v in config.items():
            if k in ["data_base_path", "paths"]:
                continue
            if isinstance(v, Box):
                self._convert_paths_inner(v, path_dict)
            elif isinstance(v, str):
                if k.endswith("_path") or k.endswith("_file"):
                    formatted = v.format(**path_dict)
                    config[k] = pathlib.Path(formatted)
        return

    def _convert_range(self, config: Box):
        for k, v in config.items():
            if isinstance(v, Box):
                if len(v.keys()) == 2 and "start" in v and "end" in v:
                    config[k] = list(range(v.start, v.end + 1))
                else:
                    self._convert_range(v)
        return
