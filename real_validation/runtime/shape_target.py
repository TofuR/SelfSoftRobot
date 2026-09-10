"""Explicit material-node correspondence for whole, tip and segment targets."""
import numpy as np


def target_indices(n_nodes, indices=None):
    if indices is None:
        return np.arange(1, n_nodes)
    values = np.asarray(indices)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
            or np.any(values < 1) or np.any(values >= n_nodes)
            or np.any(np.diff(values) != 1)):
        raise ValueError('目标节点必须是连续且有序的非 base 节点')
    return values.astype(int, copy=True)


def segment_projection(n_nodes, indices, count=32):
    """Sample the entire predicted polyline, including between material nodes."""
    ids = target_indices(n_nodes, indices)
    if len(ids) < 2 or not 2 <= count <= 128:
        raise ValueError('曲线匹配至少需要两个节点和两个采样点')
    positions = np.linspace(ids[0], ids[-1], count)
    lo = np.floor(positions).astype(int)
    hi = np.minimum(lo+1, ids[-1])
    fraction = positions-lo
    matrix = np.zeros((count, n_nodes))
    matrix[np.arange(count), lo] += 1-fraction
    matrix[np.arange(count), hi] += fraction
    return matrix


def validate_projection(matrix, samples, n_nodes, indices):
    matrix, samples = np.asarray(matrix, dtype=float), np.asarray(samples, dtype=float)
    ids = target_indices(n_nodes, indices)
    if (matrix.ndim != 2 or matrix.shape[1] != n_nodes or not 2 <= len(matrix) <= 128
            or samples.shape != (len(matrix), 2) or not np.isfinite(samples).all()
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=1), 1)
            or np.any(matrix[:, np.setdiff1d(np.arange(n_nodes), ids)] != 0)):
        raise ValueError('局部曲线采样映射无效')
    return matrix, samples


def target_distances(prediction, goal, indices=None, matrix=None, samples=None):
    goal = np.asarray(goal, dtype=float)
    if goal.shape != np.shape(prediction) or not np.isfinite(goal).all():
        raise ValueError('目标必须提供有限的完整节点数组及明确的受约束节点')
    ids = target_indices(len(goal), indices)
    if matrix is not None:
        matrix, samples = validate_projection(matrix, samples, len(goal), ids)
        return np.linalg.norm(matrix @ prediction - samples, axis=-1)
    return np.linalg.norm(np.asarray(prediction)[ids] - goal[ids], axis=-1)
