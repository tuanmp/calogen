import numpy as np


def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows with internal sort/unsort for I/O speed."""
    if len(indices) == 0:
        return dataset[:0]

    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.size == 1:
        return np.asarray(dataset[ordered], dtype=np.float32)

    if np.all(ordered[1:] >= ordered[:-1]):
        return np.asarray(dataset[ordered], dtype=np.float32)

    sort_order = np.argsort(ordered)
    sorted_indices = ordered[sort_order]
    data = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse = np.argsort(sort_order)
    return data[inverse]