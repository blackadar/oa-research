import numpy as np


def mask(arr, voxel_size=0.22):
    """
    Calculates the volume of a slice (or series) based on a voxel size.
    :param arr:
    :param voxel_size:
    :return:
    """
    assert type(arr) is np.ndarray, "Array must be an n-dimensional numpy array."
    assert arr.dtype == np.bool or arr.dtype == bool, f"Array must contain boolean values."
    assert arr.ndim == 2 or arr.ndim == 3, "Array should be (slice * ) row * column."

    return np.multiply(voxel_size, np.sum(arr))
