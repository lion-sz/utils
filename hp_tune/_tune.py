import importlib
import typing
import pathlib

from rich.console import Console
from rich.table import Table
from box import Box
import polars as pl


from app import app, state, format_float as pf
from .space import Space
from .models import hyperband, bohb, random_search


def tune(config: Box, model: typing.Callable, trace_file: pathlib.Path | None):
    """Tune a Model

    The config is a box and is expected to contain the following keys:

        * func_builder: The module that contains the function which is optimized.
        * model: The tuning model. For now only hyperband is implemented.
        * model_params: The parameter of the tuning model.
        * trace: Whether and where to write the trace file.
        * space: A dict describing the space over which to search.
        * additional_args (optional): Additional arguments passed to the model function.

    The function builder block is only required when running through the cli.
    Otherwise, pass the function to optimize directly to this method.
    The function builder method is passed the entire config and all additional args
    passed in the command line and should return a model function
    that is then given to this function.

    An example of the tune config is below.

    .. code-block:: yaml

        func_builder:
            module: pyglove.tune
            function: build_model

        model: hyperband
        model_params:
          R: 18
          eta: 2.5
          warm_start: true

        trace:
          write: true
          trace_file: "./trace.parquet"

        space:
          range: [0.1, 1.0]
          lower: [-0.5, 0.5]
          b_range: [0.1, 1.0]
          learning_rate: [0.005, 0.25]

        additional_args:
          k: 256
          gpu: true

    The model is a function that takes three arguments:

    :Parameters:
        * **point (dict)** - A dictionary of that describes the point.
          This is made up of a point in the space and
          the additional args defined in the config.
        * **resources (int)** - The number of resources (e.g. epochs)
          that the model should be trained with.
        * **warm_start (optional)** - The warm start parameters.
          If the tuning is not done with warm start, this parameter is not provided
          and therefore the model does not have to implemenet it.
    :Returns:
        The loss and - if warm start data was provided - the warm start
        data required to continue training from this point.

    This function returns a list of all points evaluated.

    Args:
        config: The (sub) config that describes the space and the
            parameters used in tuning.
        model: The model function.
        trace_file: If provided, write a trace of the optimization process
            to a file in this location.
            If None is provided, use the config settings.

    Returns:
        A list of the optimal points in the space and the trace dataframe.
    """
    space = Space.from_config(config.space)

    params = config.model_params
    if config.additional_args is not None:
        additional_args = config.additional_args.to_dict()
    else:
        additional_args = {}
    if config.model == "hyperband":
        best, trace = hyperband(
            model,
            space,
            params.R,
            params.eta,
            params.warm_start,
            additional_args,
        )
    elif config.model == "bohb":
        best, trace = bohb(
            model,
            space,
            params.R,
            params.eta,
            params.warm_start,
            additional_args,
        )
    elif config.model == "random":
        best, trace = random_search(model, space, params.R, params.n, additional_args)
    else:
        raise NotImplementedError(f"Tuning Method {config.model} is not implemented.")

    # Compute the trace dataframe.
    res = [dict(id=id(t), loss=t.loss, r=t.resources) | t.to_dict() for t in trace]
    dat = pl.from_records(res)
    # Write to trace file.
    if trace_file is None:
        if config.trace.write:
            trace_file = config.trace.trace_file
    if trace_file:
        dat.write_parquet(trace_file)
    return best, trace


@app.command("tune")
def tune_typer(
    subconfig: str,
    output: bool = True,
    trace_file: str | None = None,
    model_args: list[str] = [],
):
    config = state["config"]
    tune_config = config[subconfig]
    builder_conf = tune_config.func_builder
    mod = importlib.import_module(builder_conf.module)
    model = getattr(mod, builder_conf.function)(config, *model_args)
    best, trace = tune(tune_config, model, trace_file)

    console = Console()
    if output:
        cols = best[0]._space.dimensions
        table = Table("Loss", *[c.capitalize() for c in cols])
        for b in best:
            table.add_row(pf(b.loss), *[pf(c) for c in b.point])
        console.print(table)
