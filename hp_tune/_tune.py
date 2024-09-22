from box import Box
import polars as pl

from ._hyperband import hyperband


def tune(config: Box, model):
    """Tune a Model

    The config is a box and is expected to contain the following keys:

        * model: The tuning model. For now only hyperband is implemented.
        * model_params: The parameter of the tuning model.
        * trace: Whether and where to write the trace file.
        * space: A dict describing the space over which to search.
        * additional_args (optional): Additional arguments passed to the model function.

    Note, that this is often just a single file from a larger config.
    An example of the config is below.

    .. code-block:: yaml

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

    Returns:
        A list of Points in the space.
    """
    space = config.space.to_dict()

    params = config.model_params
    best, trace = hyperband(
        model,
        space,
        params.R,
        params.eta,
        params.warm_start,
        config.additional_args.to_dict(),
    )
    print("Tuning found these candidates:")
    for b in best:
        print(f"\t{b.losses[-1]}: {b.point}")

    # Compute the trace dataframe.
    res = []
    for t in trace:
        dat = [
            dict(id=id(t), loss=t.losses[i], r=t.resources[i], **t.point)
            for i in range(len(t.losses))
        ]
        res.extend(dat)
    dat = pl.from_records(res)
    if config.trace.write:
        dat.write_parquet(config.trace.trace_file)
    return trace
