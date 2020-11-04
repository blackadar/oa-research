"""
Finds cumulative scan volumes for masks from IWFS and DESS.
Finds cumulative BML volume of Manual BML segmentation.
Then, compares abs(vol(IWFS) - vol(DESS)) to vol(Manual BML).
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np

import common.files as files
import common.imutils as iu
import analysis.volume as vol

preds_iwfs = pathlib.Path('data/run_models/iwfs')
preds_dess = pathlib.Path('data/run_models/dess')
manual_bml_masks = pathlib.Path('data/run_models/bml')


def main():

    print(f"Loading IWFS predictions from {preds_iwfs}")
    # Find IWFS Volumes
    preds_iwfs_series = files.read_image_series(preds_iwfs, ends_with="pred.png")
    preds_iwfs_series = iu.slices_to_mask(preds_iwfs_series, threshold=0.5)
    preds_iwfs_volumes = iu.visit_to_volume(preds_iwfs_series)

    preds_iwfs_vols = {}  # Key: Patient, Val: List of Volumes

    for patient, visits in preds_iwfs_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in preds_iwfs_vols.keys():
                preds_iwfs_vols[patient] = [v, ]
            else:
                preds_iwfs_vols[patient].append(v)

    print(f"Loading DESS predictions from {preds_dess}")
    # Find 3D DESS Volumes
    preds_dess_series = files.read_image_series(preds_dess, ends_with="pred.png", no_visit=True)
    preds_dess_series = iu.slices_to_mask(preds_dess_series, threshold=0.5)
    preds_dess_volumes = iu.visit_to_volume(preds_dess_series)

    preds_dess_vols = {}  # Key: Patient, Val: List of Volumes

    for patient, visits in preds_dess_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.365 * 0.456 * 0.7)
            if patient not in preds_dess_vols.keys():
                preds_dess_vols[patient] = [v, ]
            else:
                preds_dess_vols[patient].append(v)

    missing_iwfs = np.setdiff1d(np.fromiter(preds_dess_vols.keys(), dtype=int),
                               np.fromiter(preds_iwfs_vols.keys(), dtype=int))
    missing_dess = np.setdiff1d(np.fromiter(preds_iwfs_vols.keys(), dtype=int),
                                 np.fromiter(preds_dess_vols.keys(), dtype=int))
    print(f"> {len(missing_iwfs)} cases missing in IWFS, {len(missing_dess)} cases missing in DESS.")
    print("Missing IWFS Data: (Patients in DESS not in IWFS)")
    for p in sorted(missing_iwfs):
        print(f"{p} ", end='')
    print('')
    print("Missing DESS Data: (Patients in IWFS not in DESS)")
    for p in sorted(missing_dess):
        print(f"{p} ", end='')
    print('')

    print(f"Calculating IWFS - DESS")
    # Find (IWFS - DESS)  [NOTE] only uses vol[0], so ignores multiple visits!
    diff_iwfs_dess = {}  # Key: Patient, Val: Diff of Volumes
    for patient, iwfs_volumes in preds_iwfs_vols.items():
        assert len(iwfs_volumes) > 0
        if patient in preds_dess_vols.keys():
            assert len(preds_dess_vols[patient]) > 0
            diff_iwfs_dess[patient] = abs(iwfs_volumes[0] - preds_dess_vols[patient][0])

    print(f"Loading BML from {manual_bml_masks}")
    # Find Manual BML Mask Volumes
    manual_bml_series = files.read_image_series(manual_bml_masks, ends_with="mask.bmp")
    manual_bml_series = iu.slices_to_mask(manual_bml_series, threshold=0.5)
    manual_bml_volumes = iu.visit_to_volume(manual_bml_series)

    manual_bml_vols = {}  # Key: Patient, Val: List of Volumes

    for patient, visits in manual_bml_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in manual_bml_vols.keys():
                manual_bml_vols[patient] = [v, ]
            else:
                manual_bml_vols[patient].append(v)

    missing_bml = np.setdiff1d(np.fromiter(diff_iwfs_dess.keys(), dtype=int), np.fromiter(manual_bml_vols.keys(), dtype=int))
    missing_scans = np.setdiff1d(np.fromiter(manual_bml_vols.keys(), dtype=int), np.fromiter(diff_iwfs_dess.keys(), dtype=int))
    print(f"> {len(missing_bml)} cases missing BML, {len(missing_scans)} cases missing scan data.")

    print("Missing BML Data: (Patients in Scan Data without BML)")
    for p in sorted(missing_bml):
        print(f"{p} ", end='')
    print('')
    print("Missing Scan Data: (Patients in BML without Scan Data in DESS *and* IWFS)")
    for p in sorted(missing_scans):
        print(f"{p} ", end='')
    print('')

    print(f"Finding (IWFS - DESS) / BML...")
    # Compare (IWFS - DESS) to BML Masks
    ratio = {}  # Key: Patient, Val: Diff (IWFS - DESS) - Manual BML
    for patient, diff_iwfs_dess_volumes in diff_iwfs_dess.items():
        if patient in manual_bml_vols.keys():
            assert len(manual_bml_vols[patient]) > 0
            ratio[patient] = diff_iwfs_dess_volumes / manual_bml_vols[patient][0]

    print("Plotting results..")
    # Plot the results
    plt.hist(ratio.values())
    plt.title('All Cases, Ratio (IWFS - DESS) / BML')
    plt.ylabel('Number of Cases')
    plt.xlabel('Correlation Coefficient')
    plt.show()

    cut = 10
    plt.hist(sorted(ratio.values())[:-cut])
    plt.title(f'All but {cut} Cases, Ratio (IWFS - DESS) / BML')
    plt.ylabel('Number of Cases')
    plt.xlabel('Correlation Coefficient')
    plt.show()

    r_arr = np.fromiter(ratio.values(), dtype=float)
    plt.hist(r_arr[(0.0 <= r_arr) & (r_arr <= 10.0)])
    plt.title(f'All Cases Ratio (IWFS - DESS) / BML = 0.0 - 10.0')
    plt.ylabel('Number of Cases')
    plt.xlabel('Correlation Coefficient')
    plt.show()

    bins = 10
    plt.title('Volume Distribution')
    plt.hist(diff_iwfs_dess.values(), bins, alpha=0.5, label='(IWFS - DESS)')
    plt.hist([x[0] for x in manual_bml_vols.values()], bins, alpha=0.5, label='BML Volume')
    plt.xlabel('Volume Measurement')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
