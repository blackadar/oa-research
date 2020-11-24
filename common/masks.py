"""
https://github.com/barakmichaely/bone-classification/blob/master/utils/mask_generator.py
2020-11-17

J. Blackadar: As far as I can tell, the files are formatted as:
        patient id
        visit
        "LEFT" or "RIGHT" knee
        omitted
        Image X Size
        Image Y Size
        Starting Slice (Inclusive)
        Ending Slice (Inclusive)
        "Femur"
        slice_num
        threshold
        "{"
        x y1"."y2
        "}"
        "Tibia"...
"""

import typing
import scipy.ndimage as ndimage
import numpy as np


class MaskMetadata:
    case_name = ""
    case_prefix = ""
    direction = ""
    image_width = 0
    image_height = 0
    start_slice = 0
    end_slice = 0

    def __str__(self):
        return '''
        Case number: {}
        Prefix: {}
        Direction: {}
        Image width: {}
        Image Height: {}
        Start slice: {}
        End Slice: {}
        '''.format(
                self.case_name,
                self.case_prefix,
                self.direction,
                self.image_width,
                self.image_height,
                self.start_slice,
                self.end_slice
        )


class Coordinate:
    y = 0
    x1 = 0
    x2 = 0

    def __str__(self):
        return str(self.y) + " " + str(self.x1) + "." + str(self.x2)


class SliceData:
    slice_number = 0
    coordinates: typing.List[Coordinate] = []

    def __init__(self):
        self.slice_number = 0
        self.coordinates = []


def draw_slice_image(slc: SliceData, meta: MaskMetadata):
    image_array = np.zeros((meta.image_width, meta.image_height), dtype=np.uint8)

    for c in slc.coordinates:
        for idx in range(c.x1, c.x2 + 1):
            image_array[meta.image_height - c.y, idx] = 255

    return image_array


def fill_mask(image_array):
    """
    Uses binary closing to fill holes in the mask.
    :param image_array: Mask data, typically output of draw_slice_image
    :return: Closed array, scaled back to 0/255
    """
    closed = ndimage.binary_fill_holes(image_array).astype(np.uint8)
    return np.where(closed > 0, 255, closed)


def extract_slice_mask(lines: list):
    slc = SliceData()
    slc.slice_number = -1

    try:
        ln = lines.pop(0).strip()
        if ln == "Tibia":  # TODO: Read the following data in, then discard it if not necessary
            return None
        slc.slice_number = int(ln)

    except Exception as e:
        print(f"Not a Slice Number, returning. Found : {e}")
        return None

    if lines[0] == '{':  # Parsing didn't work right, don't parse this slice
        print(f"Expected open bracket, returning. Found : {lines[0]}")
        return None

    coordinate_lines = []
    _found_first_bracket = False
    _found_last_bracket = False

    # Extract raw lines
    while not _found_last_bracket:
        if len(lines) == 0:
            break

        # Remove line from array
        line = lines.pop(0)

        # Find first line with bracket to start parsing
        if line.strip() == "{":
            _found_first_bracket = True
            continue

        if not _found_first_bracket:
            continue

        # Find last bracket to stop
        if line.strip() == "}":
            _found_last_bracket = True
            break

        # Start parsing coordinates
        coordinate_lines.append(line.strip())

    # Get coordinates from lines
    for line in coordinate_lines:
        slc.coordinates.extend(_extract_coordinates(line))

    return slc


def _extract_coordinates(line):
    # Logic:
    #   1. Extract first number as y
    #   2. For every pair after first number, create separate coordinate using the same y
    values = []

    # Get coordinate groups
    pairs: list = line.split(" ")

    y = int(pairs[0])  # Define y value
    pairs.remove(pairs[0])

    # Create coordinate from each group
    for p in pairs:
        x1, x2 = p.split(".")
        c = Coordinate()
        c.y = y - 3  # Offset of 3 is due to uneven mask (448, 444) and zero-based indexing
        c.x1 = int(x1) - 1  # Zero-based indexing
        c.x2 = int(x2) - 1  # Zero-based indexing
        values.append(c)

    return values


# Gets file metadata
def extract_mask_metadata(lines: list):
    meta = MaskMetadata()
    meta.case_name = lines.pop(0).strip()
    meta.case_prefix = lines.pop(0).strip()
    meta.direction = lines.pop(0).strip()
    lines.pop(0).strip()  # Skip this next line (not sure why)
    meta.image_width = int(lines.pop(0).strip())
    meta.image_height = int(lines.pop(0).strip())
    meta.start_slice = int(lines.pop(0).strip())
    meta.end_slice = int(lines.pop(0).strip())
    lines.pop(0).strip()  # <- "Femur" line
    return meta
