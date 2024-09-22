import typing
from math import log, ceil, floor
import random
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class Point:
    point: dict
    losses: list[float]
    resources: list[int]
    warm_start: dict | None


def sample_points(n: int, space: dict) -> list[Point]:
    points = []
    for _ in range(n):
        point = {}
        for k, dim in space.items():
            if isinstance(dim, (int, float, str)):
                # Singleton, pick this.
                point[k] = dim
            else:
                point[k] = random.uniform(dim[0], dim[1])
        points.append(Point(point, [], [], None))
    return points


def top_k(points: list[Point], k: int) -> list[Point]:
    order = sorted(range(len(points)), key=lambda i: points[i].losses[-1])
    return [points[i] for i in order[:k]]


def hyperband_est_resources(R, eta, warm_start: bool = False, verbose=False) -> int:
    """Estimate the Resources Used in a Hyperband Run

    Note, that due to rounding there is no easy expression for the total resources
    used by hyperband. Therefore, I've written this short helper function.

    Args:
        R: The maximal resources used for any given model.
        eta: Controls how quickly the resources per model
            (and thus also the number of models) decrease.
        warm_start (default False): Whether models support warm start.
        verbose (default False): Whether to print what would be done in each step.

    Returns: The total number of resources used by hyperband.
    """
    s_max = int(log(R, eta))
    B = (s_max + 1) * R
    actual = 0
    for s in range(s_max, -1, -1):
        r_last = 0
        n = ceil(B / R * eta**s / (s + 1))
        r = R * eta ** (-s)
        if verbose:
            print(f"Iteration {s} starting with {n} points")
        for i in range(s + 1):
            n_i = floor(n * eta ** (-i))
            r_i = ceil(r * eta**i)
            n_new = max(floor(n_i / eta), 1)
            if warm_start:
                actual += int(n_i * (r_i - r_last))
            else:
                actual += int(n_i * r_i)
            if verbose:
                print(
                    f"\tPicking from {n_i} models trained with {r_i - r_last * warm_start} the top {n_new}."
                )
            r_last = r_i
    return actual


def hyperband(
    model: typing.Callable,
    space: dict,
    R: int,
    eta: float,
    warm_start: bool = False,
    additional_args: dict = {},
) -> tuple[list[Point], list[Point]]:
    """Implements the Hyperband Parameter Tuning.

    The approach is described in
    `Li et al. (2016) <https://doi.org/10.48550/arXiv.1603.06560>`_
    and involves basically drawing a sample of random points, training them for some
    epochs before picking the best and continuing this loop until only on
    model is trained with the full resources `R`.

    There is always a trade-off between starting with many models, but potentially
    discarding the first before these are trained enough to distinguish noise from
    actual better performance.
    Therefore, the algorithm tries a couple different brackets that correspond
    to different amounts of resources allowed before the first models are discarded.

    This algorithm supports warm start.
    The benefits are usually good, but not incredible.
    Note, that potentially up to `R` models are evaluated in one step, and thus
    up to `R` models might be held in memory.

    Estimating the amount of resources used in hyperband is hard.
    Generally it should be around ``R * (log(R, eta) + 1)^2``, but due to rounding
    the actual number can deviate quite strongly.
    There is a helper function
    :func:`hp_tune.hyperband_est_resources` to give a precise number.

    Args:
        model: The model function.
        space: A dict that describes the space to optimize over.
        R: The maximal resources used for any given model.
        eta: Controls how aggressively the halving is.
        warm_start (default False): Whether models support warm start.
        additional_args: Additional arguments to pass to the model.

    Returns:
        A tuple of the best models from each bracket and the complete trace.
    """
    total = hyperband_est_resources(R, eta)
    t = tqdm(total=total)

    s_max = int(log(R, eta))
    B = (s_max + 1) * R
    trace: list[Point] = []
    best = []
    for s in range(s_max, -1, -1):
        n = ceil(B / R * eta**s / (s + 1))
        r = R * eta ** (-s)
        points = sample_points(n, space)
        trace.extend(points)
        r_last = 0
        for i in range(s + 1):
            n_i = floor(n * eta ** (-i))
            r_i = ceil(r * eta**i)
            n_i1 = max(floor(n_i / eta), 1)
            for point in points:
                if warm_start:
                    loss, w = model(
                        point.point | additional_args,
                        r_i - r_last,
                        point.warm_start,
                    )
                    point.warm_start = w
                else:
                    loss = model(point.point | additional_args, r_i)
                point.losses.append(loss)
                point.resources.append(r_i)
                t.update(r_i)
            points = top_k(points, n_i1)
            r_last = r_i
        if len(points) > 1:
            points = top_k(points, 1)
        best.append(points[0])
    t.close()
    return best, trace
