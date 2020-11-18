import collections
from typing import Union

import numpy as np
from pydicom import Dataset


def update_nested_dict(original: dict, update: Union[dict, collections.abc.Mapping]):
    """ Updates a nested dictionary using another (sparse) nested dictionary.
    Imitates dict.update(), adding value if it doesn't exist or just modifying it.
    :param original: Dictionary to update
    :param update: Updates for Dictionary, as a Dictionary (can be sparse)
    :return: Updated Dictionary
    """
    # Iterate over each current-level item
    for k, v in update.items():
        # If the value is a Mapping, it's a next level which should be handled recursively
        if isinstance(v, collections.abc.Mapping) and not isinstance(v, Dataset):
            original[k] = update_nested_dict(original.get(k, {}), v)
        # Otherwise, it's the end of a recursive tree
        else:
            original[k] = v
    return original


def find_diff(i1, i2):
    i2_missing = []
    i1_missing = []

    for item in i1:
        if item not in i2:
            i2_missing.append(item)

    for item in i2:
        if item not in i1:
            i1_missing.append(item)

    return i1_missing, i2_missing


def slices_to_mask(nested_dict: dict, threshold=0.5):
    """
    Converts image data to boolean. Rounds if neccessary.
    Note: expects typical nested dict! NOT modified to be volumes, etc.
    :param threshold: Threshold for True value
    :param nested_dict: Typical nested dict, ex. parsed by common.files.read_bmp_series
    :return: Updated version of the dict
    """
    assert type(nested_dict) is dict, "Expecting dictionary as input."
    result = {}

    for patient, visits in nested_dict.items():
        for visit, slice_ids in visits.items():
            for slice_id, im in slice_ids.items():
                new_im = (im > threshold)
                update_nested_dict(result, {patient: {visit: {slice_id: new_im}}})
    return result


def visit_to_volume(nested_dict: dict):
    """
    Converts nested dict of {patient: {visit: {slice_id: image} } } to {patient: {visit: 3D Array } }
    :param nested_dict: Typical nested dict, ex. parsed by common.files.read_bmp_series
    :return: Updated version of the dict
    """
    assert type(nested_dict) is dict, "Expecting dictionary as input."
    result = {}

    for patient, visits in nested_dict.items():
        for visit, slice_ids in visits.items():
            visit_volume = []
            for slice_id, im in slice_ids.items():
                visit_volume.append(im)
            visit_volume = np.array(visit_volume)
            update_nested_dict(result, {patient: {visit: visit_volume}})

    return result
