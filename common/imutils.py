import numpy as np
import common.dictutils as du


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
                du.update_nested_dict(result, {patient: {visit: {slice_id: new_im}}})
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
            du.update_nested_dict(result, {patient: {visit: visit_volume}})

    return result
