from dataclasses import dataclass

import numpy as np
from box import Box, BoxList


@dataclass
class Space:
    """The space over which the optimization should run.

    Note, that the ordering of dimensions in the space is potentially
    different from the order they appear in the original config.

    For now only supports continuous dimensions.
    """

    dimensions: list[str]
    types: list[str]
    cont_ind: list[int]
    cont_lower: np.ndarray[np.float64]
    cont_upper: np.ndarray[np.float64]

    @classmethod
    def from_config(cls, config: Box):
        names = []
        types = []
        cont_ind = []
        lower = []
        upper = []
        for i, (name, dim) in enumerate(config.items()):
            names.append(name)
            if isinstance(dim, BoxList):
                # Continuous dimension
                cont_ind.append(i)
                types.append(dim)
                lower.append(dim[0])
                upper.append(dim[1])
        return Space(names, types, cont_ind, lower, upper)

    def check_contains_points(self, points: np.ndarray):
        if not points.ndim == 2:
            raise NotImplementedError("Check contains points called with not 2d array.")
        res = np.zeros(shape=points.shape[0], dtype=np.int64)
        for i, ind in enumerate(self.cont_ind):
            res += points[:, ind] < self.cont_lower[i]
            res += points[:, ind] > self.cont_upper[i]
        return res == 0

    def clip_points(self, points: np.ndarray):
        if not points.ndim == 2:
            raise NotImplementedError("Check contains points called with not 2d array.")
        res = np.zeros(shape=points.shape[0], dtype=np.int64)
        for i, ind in enumerate(self.cont_ind):
            temp = points[:, ind] < self.cont_lower[i]
            points[temp, ind] = self.cont_lower[i]
            res += temp
            temp = points[:, ind] > self.cont_upper[i]
            points[temp, ind] = self.cont_upper[i]
            res += temp
        print("Share, num of points inside the space: ", np.mean(res == 0), np.sum(res == 0))
        return points



class Point:
    _space: Space
    point: np.ndarray
    loss: float | None
    resources: int | None

    def __init__(
        self, space: Space, point: np.ndarray, loss: float = None, resources: int = None
    ):
        self._space = space
        self.point = point
        self.loss = loss
        self.resources = resources

    def __eq__(self, other) -> bool:
        return all(self.point == other.point)

    def with_loss(self, loss: float, resource: int) -> "Point":
        return Point(self._space, self.point, loss, resource)

    @classmethod
    def from_np(cls, space: Space, arr: np.ndarray):
        if arr.ndim == 1:
            return cls(space, arr)
        points = []
        for i in range(np.shape(arr)[0]):
            points.append(cls(space, arr[i, :]))
        return points

    def to_dict(self):
        return dict(zip(self._space.dimensions, [float(v) for v in self.point]))


def sample_uniform(space: Space, n: int) -> list[Point]:
    """Sample points uniformly from the given space."""
    points = np.zeros(shape=(n, len(space.dimensions)))
    # For now only continuous dimensions, so this is ok.
    for i, ind in enumerate(space.cont_ind):
        points[:, ind] = np.random.uniform(space.cont_lower[i], space.cont_upper[i], n)
    return Point.from_np(space, points)
