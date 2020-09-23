from PIL import Image
import pathlib
import logging
import numpy as np
import common.dictutils as du
log = logging.getLogger(__name__)


def read_image_series(path, ends_with="pred.png"):
    """
    Reads image files in series, following name format:
    {Patient ID}_v{Visit ID}_{Slice Number}
    Returns the pixel data arrays in a nested dict by patient and visit.
    :param ends_with: String ending of filename (before ending) to match
    :param path: Path to folder containing .bmps
    :return: Nested dictionary {patient: {visit: {slice_id: image} } }
    """
    result = {}
    folder = pathlib.Path(path)
    files = list(folder.glob(f"*{ends_with}"))
    if len(files) < 1:
        log.error(f"No images (ending in '{ends_with}') found in {folder}!")
        exit(-1)

    for idx, file in enumerate(files):
        name_split = file.name[:-1*len(ends_with)].split('_')[:-1]  # Ignore .bmp, store contents between underscores
        if len(name_split) != 3:
            log.warning(f'File "{file.name}" does not conform to the naming standard.')
            continue
        (patient, visit, slice_id) = name_split
        im = np.array(Image.open(file))
        du.update_nested_dict(result, {patient: {visit: {slice_id: im}}})

    return result
