"""Shared pressure bounds and full-horizon action parameterization."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ActionBounds:
    lower: np.ndarray
    upper: np.ndarray
    rise: np.ndarray  # per model step, already normalized
    fall: np.ndarray

    def __post_init__(self):
        arrays = [np.asarray(v) for v in (self.lower, self.upper, self.rise, self.fall)]
        if (any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays)
                or any(not np.isfinite(a).all() for a in arrays)
                or np.any(arrays[0] > arrays[1])
                or np.any(arrays[2] < 0) or np.any(arrays[3] < 0)):
            raise ValueError("invalid action bounds or rate units")

    def project(self, actions, previous):
        result = np.asarray(actions, dtype=np.float64).copy()
        previous = np.asarray(previous, dtype=np.float64).copy()
        if not np.isfinite(result).all() or not np.isfinite(previous).all():
            raise ValueError("nonfinite pressure")
        if np.any(previous < self.lower-1e-7) or np.any(previous > self.upper+1e-7):
            raise ValueError("initial pressure outside bounds")
        for i in range(len(result)):
            result[i] = np.clip(result[i], np.maximum(self.lower, previous-self.fall),
                                np.minimum(self.upper, previous+self.rise))
            previous = result[i]
        return result

    def with_rate_fraction(self, fraction):
        """Planning-only slew budget; physical pressure/rate limits stay intact."""
        if isinstance(fraction,(bool,np.bool_)) or not np.isfinite(fraction) or not 0<float(fraction)<=1:
            raise ValueError('规划速度比例必须在 (0, 1]')
        return ActionBounds(self.lower.copy(),self.upper.copy(),
                            self.rise*float(fraction),self.fall*float(fraction))

    def valid(self, actions, previous, tolerance=2e-6):
        actions = np.asarray(actions)
        delta = np.diff(np.vstack([previous, actions]), axis=0)
        return bool(np.isfinite(actions).all() and
                    np.all(actions >= self.lower-tolerance) and np.all(actions <= self.upper+tolerance) and
                    np.all(delta <= self.rise+tolerance) and np.all(delta >= -self.fall-tolerance))


def block_basis(horizon, channels, blocks):
    """Every future action is covered, including the final row."""
    count = min(horizon, blocks)
    assignment = np.minimum(np.arange(horizon)*count//horizon, count-1)
    return np.kron(np.eye(count)[assignment], np.eye(channels))
