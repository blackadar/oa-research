"""
Finds volume measurements from two datasets.
Correlates the patients,
and finds the difference.
"""

import pathlib
import common.files as files
import common.imutils as iu
import analysis.volume as vol

preds_a = pathlib.Path('data/preds/')
preds_b = pathlib.Path('data/preds/')


def main():
    preds_a_series = files.read_image_series(preds_a, ends_with="pred.png")
    preds_a_series = iu.slices_to_mask(preds_a_series, threshold=0.5)
    preds_a_volumes = iu.visit_to_volume(preds_a_series)

    preds_a_vols = {}  # Key: Patient, Val: List of Volumes

    for patient, visits in preds_a_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in preds_a_vols.keys():
                preds_a_vols[patient] = [v, ]
            else:
                preds_a_vols[patient].append(v)

    preds_b_series = files.read_image_series(preds_b, ends_with="pred.png")
    preds_b_series = iu.slices_to_mask(preds_b_series, threshold=0.5)
    preds_b_volumes = iu.visit_to_volume(preds_b_series)

    preds_b_vols = {}  # Key: Patient, Val: List of Volumes

    for patient, visits in preds_b_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in preds_b_vols.keys():
                preds_b_vols[patient] = [v, ]
            else:
                preds_b_vols[patient].append(v)

    # TODO: Compare Volumes. Average? (Mean, Median)


if __name__ == "__main__":
    main()
