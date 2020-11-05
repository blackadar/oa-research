import collections
from typing import Union


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
        if isinstance(v, collections.abc.Mapping):
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
