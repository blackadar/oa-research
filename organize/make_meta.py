"""
These functions help make the meta directory in each visit folder:
{Patient ID}/
    {Visit ID}/
        meta/
            images/ : Images, as uint8 png converted from DICOM
            bml_masks/ : BML masks, as png converted from .txt files
            bone_masks/ : Bone Segmentation, as png converted from .txt files
            bone_segmented_images/ : Images, with bone_masks applied

"""
import common.files as files
import common.masks as masks
import pathlib
import sys
import numpy as np


def make_meta_for_patients(path, images=True, bone_segmented_images=True, bml=True, bone=True):
    """
    Function to handle iterating through patients and visits, then generate desired meta folders.
    Will overwrite existing files with updated versions, but will not delete other files in the folders.
    :param bone_segmented_images: Generate images only in regions marked as bone, by bone masks
    :param bone: bool, Generate bone masks
    :param bml: bool, Generate BML masks
    :param images: bool, Generate png images
    :param path: pathlib.Path or str, Path to the folder containing Patients (Phase3, Phase4, etc)
    :return: None
    """
    context = {}
    folder = pathlib.Path(path)
    patients = [patient for patient in folder.iterdir() if patient.is_dir()]

    for idx, patient in enumerate(patients):
        sys.stdout.write(f"\rProcessing {idx} of {len(patients)} patients from {path}. ({idx / len(patients) * 100: .2f}%)\r\n")
        sys.stdout.flush()
        context['patient_key'] = int(patient.name)
        visits = [visit for visit in patient.iterdir() if visit.is_dir()]
        for visit in visits:
            context['visit_key'] = visit.name

            if images:
                make_meta_images(visit, visit / 'meta' / 'images', context=context)
            if bml:
                make_meta_bml_masks(visit, visit / 'meta' / 'bml_masks', context=context)
            if bone:
                make_meta_bone_masks(visit, visit / 'meta' / 'bone_masks', context=context)
            if bone_segmented_images:
                make_meta_bone_segmented_images(visit, visit / 'meta' / 'bone_segmented_images', context=context)

    sys.stdout.write(f"\rProcessed {len(patients)} patients from {path}.\r\n")
    sys.stdout.flush()


def make_meta_images(input_path, output_path, context=None):
    """
    Creates images from DICOMS at input_path, puts output images at output_path.
    :param context: dict, keys 'patient_key' and 'visit_key' to use in output filename
    :param input_path: pathlib.Path or str, path to folder containing input DICOMs
    :param output_path: pathlib.Path or str, path to folder to contain output .pngs
    :return: None
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    slices = list(input_path.glob('*[!.txt]'))
    if context:
        patient = context['patient_key']
        visit = context['visit_key']
    else:
        patient = 'x'
        visit = 'vxx'
    for slc in slices:
        if not slc.is_dir():
            slc_num = slc.name
            files.write_image(files.read_dicom(slc), output_path=output_path / f'{patient}_{visit}_{slc_num}.bmp')


def make_meta_bml_masks(input_path, output_path, fill_holes=False, context=None):
    """
    Creates masks from BML txt at input_path, puts output masks at output_path.
    :param fill_holes: Use Binary Closure to Fill Mask Holes
    :param context: dict, keys 'patient_key' and 'visit_key' to use in output filename
    :param input_path: pathlib.Path or str, path to folder containing input txt file
    :param output_path: pathlib.Path or str, path to folder to contain output .png masks
    :return: None
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if context:
        patient = context['patient_key']
        visit = context['visit_key']
    else:
        patient = 'x'
        visit = 'vxx'

    # Find the BML mask file:
    bml_files = list(input_path.glob('BML*.txt'))
    if len(bml_files) > 1:
        print(f"{input_path} has {len(bml_files)} files matching BML pattern: {bml_files}")
    elif len(bml_files) == 0:
        print(f"{input_path} has no files matching BML pattern.")
        return
    bml_text = bml_files[0]

    # Parse the BML file
    slices = files.read_masks_from_txt(bml_text)

    # If desired, fill in the holes in the masks
    if fill_holes:
        new_slices = {}
        for k, v in slices.items():
            new_slices[k] = masks.fill_mask(v)  # Makes one continuous region, atypical for BML but here just in case
        slices = new_slices

    # Write a mask for each slice
    # TODO: Write blank mask if data exists but not BML?
    for slc_num, slc in slices.items():
        files.write_mask(slc, output_path/f'{patient}_{visit}_{slc_num}_mask.bmp')


def make_meta_bone_masks(input_path, output_path, fill_holes=True, context=None):
    """
    Creates masks from Bone txt at input_path, puts output masks at output_path.
    :param fill_holes: Use Binary Closure to Fill Mask Holes
    :param context: dict, keys 'patient_key' and 'visit_key' to use in output filename
    :param input_path: pathlib.Path or str, path to folder containing input txt file
    :param output_path: pathlib.Path or str, path to folder to contain output .png masks
    :return: None
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if context:
        patient = context['patient_key']
        visit = context['visit_key']
    else:
        patient = 'x'
        visit = 'vxx'

    # Find the Bone mask file:
    bone_files = list(input_path.glob('*FemurBone.txt'))
    if len(bone_files) > 1:
        print(f"{input_path} has {len(bone_files)} files matching Bone pattern: {bone_files}")
    elif len(bone_files) == 0:
        print(f"{input_path} has no files matching Bone pattern.")
        return
    bone_text = bone_files[0]

    # Parse the Bone file
    slices = files.read_masks_from_txt(bone_text)

    # If desired, fill in the holes in the masks
    if fill_holes:
        new_slices = {}
        for k, v in slices.items():
            new_slices[k] = masks.fill_mask(v)  # Makes one continuous region, for the bone label
        slices = new_slices

    # Write a mask for each slice
    for slc_num, slc in slices.items():
        files.write_mask(slc, output_path / f'{patient}_{visit}_{slc_num}_mask.bmp')


def make_meta_bone_segmented_images(input_path, output_path, fill_holes=True, context=None):
    """
    Generates images with only bone region, using DICOMs and txt masks in the input_path.
    Outputs images to output_path.
    :param fill_holes:
    :param context:
    :param input_path:
    :param output_path:
    :return:
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    image_slices = list(input_path.glob('*[!.txt]'))

    # Find the Bone mask file:
    bone_files = list(input_path.glob('*FemurBone.txt'))
    if len(bone_files) > 1:
        print(f"{input_path} has {len(bone_files)} files matching Bone pattern: {bone_files}")
    elif len(bone_files) == 0:
        print(f"{input_path} has no files matching Bone pattern.")
        return
    bone_text = bone_files[0]

    # Parse the Bone file
    bone_mask_slices = files.read_masks_from_txt(bone_text)

    # If desired, fill in the holes in the masks
    if fill_holes:
        new_slices = {}
        for k, v in bone_mask_slices.items():
            new_slices[k] = masks.fill_mask(v)  # Makes one continuous region, for the bone label
        bone_mask_slices = new_slices

    if context:
        patient = context['patient_key']
        visit = context['visit_key']
    else:
        patient = 'x'
        visit = 'vxx'

    for slc in image_slices:
        if not slc.is_dir():
            try:
                slc_num = int(slc.name)
            except TypeError as e:
                continue
            if slc_num in bone_mask_slices.keys():
                # We need the image and mask the same size - (448, 448)
                im = np.pad(files.read_dicom(slc).pixel_array, [(0, 4), (0, 0)])  # Images are (444, 448)
                mask = np.pad(bone_mask_slices[slc_num].astype(bool), [(0, 0), (0, 4)])  # Masks are (448, 444)
                im[~mask] = 0  # This applies the mask to the image - setting all positions without mask to zero
                files.write_image(im, output_path=output_path / f'{patient}_{visit}_{slc_num}.bmp')
