from typing import Annotated

import typer
from typer import Option
import sys

from app_template import ConfigReader


def setup_callback(app):
    # @app.callback(["name", "config", "config_overwrite"])
    @app.callback()
    def load_config(
        name: str | None = None,
        config: list[str] = ["./config"],
        disable_cache: bool = False,
        k: Annotated[list[str], Option(help="keys to overwrite. Provide also v")] = [],
        v: Annotated[list[str], Option(help="Values for overwrites")] = [],
    ):
        if len(config) == 0:
            typer.echo("No config provided. Exiting...")
            sys.exit(1)
        assert len(k) == len(v), "Unequal number of keys and values."
        overwrites = dict(zip(k, v))
        config = ConfigReader(name).load(config[0], overwrites)
        if disable_cache:
            config.use_cache = False
        sys.modules["app_template"].config = config
        return
