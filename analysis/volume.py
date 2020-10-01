import numpy as np


def mask(arr, voxel_size=1):
    """
    Calculates the volume of a slice (or series) based on a voxel size.
    :param arr: np.ndarray of type bool
    :param voxel_size: Size of a voxel, in your unit of choice
    :return: Single value, total volume (slice or volume-wise)
    """
    assert type(arr) is np.ndarray, "Array must be an n-dimensional numpy array."
    assert arr.dtype == np.bool or arr.dtype == bool, f"Array must contain boolean values."
    assert arr.ndim == 2 or arr.ndim == 3, "Array should be (slice * ) row * column."

    return np.multiply(voxel_size, np.count_nonzero(arr))
