import typer
import sys

from config_reader import ConfigReader

app = typer.Typer()
state = {}


def format_float(f: float) -> str:
    return "{:.3f}".format(f)
    # return "{:4.2E}".format(Decimal(f))


@app.callback(["name", "config", "config_overwrite"])
def load_config(name: str | None = None, config: list[str] = ["./config"]):
    if len(config) == 0:
        typer.echo("No config provided. Exiting...")
        sys.exit(1)
    config = ConfigReader(name).load(config[0])
    state["config"] = config
    return
