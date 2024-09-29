import typing
from math import log, ceil, floor
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from .space import Space, Point, sample_uniform


def random_search(
    model: typing.Callable,
    space: Space,
    R: int,
    n: int,
    additional_args: dict = {},
) -> tuple[list[Point], list[Point]]:
    """Perform a random search over the space.

    The function evaluates n points with resources R per point.

    Returns: The five best points and a full trace (less if n < 5).
    """
    t = tqdm(total=n * R)
    points = sample_uniform(space, n)
    trace = []
    for point in points:
        loss = model(point.to_dict() | additional_args, R)
        trace.append(point.with_loss(loss, R))
        t.update(R)
    return sorted(trace, key=lambda p: p.loss)[: min(n, 5)], trace


def top_k(points: list[Point], k: int) -> list[Point]:
    return sorted(points, key=lambda p: p.loss)[:k]


def _successive_halving(
    points: list[Point],
    model,
    s: int,
    eta: float,
    n: float,
    r: float,
    warm_start: bool,
    additional_args: dict,
    t,
) -> tuple[Point, list[Point]]:
    r_last = 0
    trace = []
    for i in range(s + 1):
        trained = []
        n_i = floor(n * eta ** (-i))
        r_i = ceil(r * eta**i)
        n_i1 = max(floor(n_i / eta), 1)
        for point in points:
            if warm_start:
                raise NotImplementedError("Currently disabled.")
                loss, w = model(
                    point.to_dict() | additional_args,
                    r_i - r_last,
                    point.warm_start,
                )
                point.warm_start = w
            else:
                loss = model(point.to_dict() | additional_args, r_i)
            trained.append(point.with_loss(loss, r_i))
            t.update(r_i)
        trace.extend(trained)
        points = top_k(trained, n_i1)
        r_last = r_i
    if len(points) > 1:
        points = top_k(points, 1)
    return points[0], trace


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
    space: Space,
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
    args = (warm_start, additional_args, t)

    s_max = int(log(R, eta))
    B = (s_max + 1) * R
    trace: list[Point] = []
    best: list[Point] = []
    for s in range(s_max, -1, -1):
        n = ceil(B / R * eta**s / (s + 1))
        r = R * eta ** (-s)
        points = sample_uniform(space, n)
        bracket_best, bracket_trace = _successive_halving(
            points, model, s, eta, n, r, *args
        )
        best.append(bracket_best)
        trace.extend(bracket_trace)
    t.close()
    return best, trace


def sample_points_bohb(n, space: Space, losses, q, rho):
    # Find the first budget.
    n_min = len(space.dimensions) + 1
    if not any(len(v) >= n_min + 2 for v in losses.values()):
        return sample_uniform(space, n)
    max_b = max(b for b, v in losses.items() if len(v) >= n_min)
    # Fit the KDEs.
    selected = losses[max_b]
    x = sorted(selected, key=lambda x: x.loss)
    nb = len(selected)
    nbl = max(n_min, floor(q * nb))
    xl = np.stack([x.point for x in x[:nb]])
    xu = np.stack([x.point for x in x[-(nb - nbl) :]])
    kde_l = KernelDensity(kernel="gaussian", bandwidth="scott").fit(xl)
    kde_u = KernelDensity(kernel="gaussian", bandwidth="scott").fit(xu)

    points = []
    # First sample the random points.
    n_rand = np.random.binomial(n, rho)
    points.extend(sample_uniform(space, n_rand))
    # Sample a point from the kernel densities.
    candidates = []
    i = 0
    while i < 10:
        temp = kde_l.sample((n - n_rand) * 5000)
        valid = space.check_contains_points(temp)
        temp = temp[valid, :]
        candidates.append(temp)
        i += 1
        if sum(map(len, candidates)) > ((n - n_rand) * 10):
            break
    if i == 10:
        print("Failed to sample enough points in the space for b {max_b}.")
        # print("Exiting.")
        # import sys; sys.exit(1)
        print(f"Returning all random points for b {max_b}.")
        return sample_uniform(space, n)
    candidates = np.concatenate(candidates)
    # The scores returned are log scores, so transform to normal.
    ratio = np.exp(kde_l.score_samples(candidates) - kde_u.score_samples(candidates))
    ord = sorted(range(len(candidates)), key=lambda i: ratio[i], reverse=True)
    # Pick the best points and append.
    temp = Point.from_np(space, candidates[ord[: (n - n_rand)], :])
    points.extend(temp)
    return points


def bohb(
    model: typing.Callable,
    space: Space,
    R: int,
    eta: float,
    warm_start: bool = False,
    additional_args: dict = {},
) -> tuple[list[Point], list[Point]]:
    """Implements the Bayesian Hyperband Parameter Tuning."""
    total = hyperband_est_resources(R, eta)
    t = tqdm(total=total)
    args = (warm_start, additional_args, t)

    s_max = int(log(R, eta))
    B = (s_max + 1) * R
    trace: list[Point] = []
    losses = defaultdict(list)
    best = []
    for s in range(s_max, -1, -1):
        n = ceil(B / R * eta**s / (s + 1))
        r = R * eta ** (-s)
        points = sample_points_bohb(n, space, losses, 0.5, 0.2)
        best_point, bracket_trace = _successive_halving(
            points, model, s, eta, n, r, *args
        )
        best.append(best_point)
        trace.extend(bracket_trace)
        for point in bracket_trace:
            losses[point.resources].append(point)
    t.close()
    return best, trace
