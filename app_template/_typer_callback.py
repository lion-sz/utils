import typer
import sys

from app_template import ConfigReader


def setup_callback(app):
    # @app.callback(["name", "config", "config_overwrite"])
    @app.callback()
    def load_config(
        name: str | None = None,
        config: list[str] = ["./config"],
        disable_cache: bool = False,
    ):
        if len(config) == 0:
            typer.echo("No config provided. Exiting...")
            sys.exit(1)
        config = ConfigReader(name).load(config[0])
        if disable_cache:
            config.use_cache = False
        sys.modules["app_template"].config = config
        return
