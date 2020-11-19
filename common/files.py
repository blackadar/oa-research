from PIL import Image
import pydicom
import pathlib
import logging
import numpy as np
import common.dictutils as du
import common.masks as mask_generator
import sys

log = logging.getLogger(__name__)


def read_dicom(path):
    """
    Reads a DICOM from path.
    For now, this wraps pydicom to allow for easier refactoring if needed.
    :param path: pathlib.Path or str, path to the DICOM file
    :return: pydicom.FileDataSet
    """
    return pydicom.dcmread(path)


def write_image(dataset, output_path, size=(448, 448)):
    """
    Writes an image array (or pydicom Dataset) in memory, to an image file.
    :param dataset: array or pydicom.DataSet with image data
    :param output_path: pathlib.Path or str, path to desired output image
    :param size: tuple (int, int) desired size (centered box crop will be applied)
    :return: None
    """
    # Grab just the pixel data if provided a pydicom.Dataset
    if isinstance(dataset, pydicom.Dataset):
        dataset = dataset.pixel_array
    # Rescale the Image to 0 -> 255
    arr = dataset / np.max(dataset)
    arr = 255 * arr
    arr = np.uint8(arr)

    if arr.shape != size:  # If specified size is different...
        if arr.shape[0] > size[0]:  # If dim 0 is bigger than specified, trim it
            x_start = arr.shape[0] // 2 - size[0] // 2
            x_end = x_start + size[0]
            arr = arr[x_start:x_end]
        else:  # Else, if it's the same or lesser, pad it  TODO:MATLAB may use left aligned. Should be consistent.
            diff = size[0] - arr.shape[0]
            arr = np.pad(arr, [(diff//2, diff - (diff//2)), (0, 0)])
        if arr.shape[1] > size[1]:  # If dim 1 is bigger than specified, trim it
            y_start = arr.shape[1] // 2 - size[1] // 2
            y_end = y_start + size[1]
            arr = arr[:, y_start:y_end]
        else:  # Else, if it's the same or lesser, pad it
            diff = size[1] - arr.shape[1]
            arr = np.pad(arr, [(0, 0), (diff // 2, diff - (diff // 2))])

    assert arr.shape == size, f"Error in logic! Shape was supposed to be {size}, but was {arr.shape}"

    im = Image.fromarray(arr)
    im.save(output_path)


def read_masks_from_txt(path):
    """
    Uses Barak's mask_generator to read masks from their .txt files.
    Depends on common.masks.
    Based on main from https://github.com/barakmichaely/bone-classification/blob/master/utils/mask_generator.py
    :param path: path-like, path to the .txt file
    :return: dict, {slice_number: PIL Image}
    """
    # This goes through the .txt, line by line and parses metadata first, then each segmented region.
    # The file format is proprietary. See masks.py for info about the structure.

    file = open(path, 'r')
    file_lines = file.readlines()
    metadata = mask_generator.extract_mask_metadata(file_lines)
    slices = []

    _got_last_slice = False
    while not _got_last_slice and len(file_lines) > 0:
        try:
            slc = mask_generator.extract_slice_mask(file_lines)
            if slc is None:
                break
            slices.append(slc)
        except Exception as e:
            print(f'Slice parse error: {e}')
            continue

        if slc.slice_number >= metadata.end_slice:
            _got_last_slice = True
            break

    data = {}
    for s in slices:
        data[s.slice_number] = mask_generator.draw_slice_image(s, metadata)

    return data


def write_mask(data, path):
    """
    Writes a PIL Image mask to file.
    :param data: PIL Image to write
    :param path: pathlib.Path or str path to write Image
    :return: None
    """
    im = Image.fromarray(data, mode='L')
    im.save(path)


def read_image_series(path, ends_with="pred.png", no_visit=False):
    """
    Reads image files in series, following name format:
    {Patient ID}_v{Visit ID}_{Slice Number}
    Returns the pixel data arrays in a nested dict by patient and visit.
    :param no_visit: If the filenames don't have v01, etc.
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
        name_split = file.name[:-1 * len(ends_with)].split('_')[:-1]  # Ignore .bmp, store contents between underscores
        if (no_visit is False and len(name_split) != 3) or (no_visit is True and len(name_split) != 2):
            log.warning(f'File "{file.name}" does not conform to the naming standard.')
            continue
        if no_visit:
            (patient, visit, slice_id) = int(name_split[0]), 'v00', int(name_split[1])
        else:
            (patient, visit, slice_id) = name_split
            patient = int(patient)
            slice_id = int(slice_id)
        im = np.array(Image.open(file))
        du.update_nested_dict(result, {patient: {visit: {slice_id: im}}})

    return result


def read_cdi_bone_volumes(path, ends_with=".txt"):
    """
    Reads patient CDI Bone Volumes from a directory of .txt files.
    Returns a dict of patient, (start, end)
    :param path: Path to folder containing .txt files
    :param ends_with: File ending to match against. Can exclude unwanted .txts
    :return: dict of patient, (start, end)
    """

    result = {}
    folder = pathlib.Path(path)
    files = list(folder.glob(f"*{ends_with}"))
    if len(files) < 1:
        log.error(f"No files (ending in '{ends_with}') found in {folder}!")
        exit(-1)

    for idx, file in enumerate(files):
        name_split = file.name[:-1 * len(ends_with)].split('_')[:-1]  # Ignore .txt, store contents between underscores
        if len(name_split) != 5:
            log.warning(f'File "{file.name}" does not conform to the naming standard, ending with {ends_with}.')
            continue
        (_, _, _, patient, visit) = name_split
        patient = int(patient)
        with open(file) as fp:
            try:
                for i, line in enumerate(fp):
                    if i == 5:
                        start = int(line)
                    elif i == 6:
                        end = int(line)
                    elif i > 6:
                        break
                du.update_nested_dict(result, {patient: {visit: (start, end)}})
            except (TypeError, ValueError) as e:
                log.error(f"Error parsing {file}: {e}")

    return result


def read_dicom_series(path):
    """
    Reads DICOM images into memory.
    Path naming: path:/{PATIENT}/{VISIT}/{SLICE_ID} (DICOM)
    Returned Dict: {patient: {visit: {slice_id: DataSet} } }
    :param path: Top-Level path containing patient directories
    :return: Dictionary, {patient: {visit: {slice_id: DataSet} } }
    """
    result = {}
    folder = pathlib.Path(path)
    patients = [patient for patient in folder.iterdir() if patient.is_dir()]

    for idx, patient in enumerate(patients):
        sys.stdout.write(f"\r Reading {idx} of {len(patients)} patients from disk. ({idx / len(patients) * 100: .2f}%)")
        sys.stdout.flush()
        patient_key = int(patient.name)
        visits = [visit for visit in patient.iterdir() if visit.is_dir()]
        for visit in visits:
            visit_key = visit.name
            slices = list(visit.glob('*'))
            for slc in slices:
                slice_key = int(slc.name)
                du.update_nested_dict(result, {patient_key: {visit_key: {slice_key: read_dicom(slc)}}})
    sys.stdout.write(f"\rRead {len(patients)} files from storage.\r\n")
    sys.stdout.flush()
    return result


def write_dicom_series_as_png(series: dict, output_path=pathlib.Path('out/'), size=(352, 352)):
    """
    Takes a DICOM series in memory, and writes them as .png files of a specific size.
    Box crops at center to fit the size.
    :param size: Two-Tuple, (x-length: int, y-length: int) desired image size
    :param output_path: Folder to place .pngs into
    :param series: Series in memory, as dict {patient: {visit: {slice_id: DataSet} } }
    :return: None
    """
    try:
        assert type(output_path) is pathlib.Path
    except AssertionError:
        try:
            output_path = pathlib.Path(output_path)
        except Exception as e:
            raise TypeError(f"'{output_path}' cannot be coerced into a path: {e}")

    for idx, (patient_id, patient) in enumerate(series.items()):
        sys.stdout.write(f"\rWriting: {idx} of {len(series)} patients to disk. ({idx / len(series) * 100: .2f}%)")
        sys.stdout.flush()
        for visit_id, visit in patient.items():
            for slice_id, slc in visit.items():
                write_image(slc, f"{patient_id}_{visit_id}_{slice_id}.png", size)
    sys.stdout.write(f"\rWrote {len(series)} files to storage.\r\n")
    sys.stdout.flush()
