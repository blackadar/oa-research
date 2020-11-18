"""
Finds cumulative scan volumes for masks from IWFS and DESS.
Finds cumulative BML volume of Manual BML segmentation.
Then, compares abs(vol(IWFS) - vol(DESS)) to vol(Manual BML).
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import common.dictutils
import common.files as files
import analysis.volume as vol

preds_iwfs = pathlib.Path('data/run_models/iwfs')
preds_dess = pathlib.Path('data/run_models/dess')
manual_bml_masks = pathlib.Path('data/run_models/bml')
cdi_files = pathlib.Path('data/cdi/OldMethod')


def main():

    print(f"Loading IWFS predictions from {preds_iwfs}")
    preds_iwfs_series = files.read_image_series(preds_iwfs, ends_with="pred.png")
    preds_iwfs_series = common.dictutils.slices_to_mask(preds_iwfs_series, threshold=0.5)
    preds_iwfs_volumes = common.dictutils.visit_to_volume(preds_iwfs_series)
    preds_iwfs_totals = {}  # Key: Patient, Val: List of Volumes

    print(f"Calculating IWFS volume metrics")
    for patient, visits in preds_iwfs_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in preds_iwfs_totals.keys():
                preds_iwfs_totals[patient] = [v, ]
            else:
                preds_iwfs_totals[patient].append(v)

    print(f"Loading DESS predictions from {preds_dess}")
    preds_dess_series = files.read_image_series(preds_dess, ends_with="pred.png")
    preds_dess_series = common.dictutils.slices_to_mask(preds_dess_series, threshold=0.5)
    preds_dess_volumes = common.dictutils.visit_to_volume(preds_dess_series)
    preds_dess_totals = {}  # Key: Patient, Val: List of Volumes

    # print(f"Loading CDI markers from {cdi_files}")
    # cdi = files.read_cdi_bone_volumes(cdi_files, ends_with="OldMethod_Femur.txt")
    # print(f"Applying {len(cdi)} patient CDI markers.")
    # print(f'CDI Markers for visits not Present in DESS:')
    # for patient, visits in cdi.items():
    #     for visit, (start, end) in visits.items():
    #         try:
    #             preds_dess_volumes[patient][visit] = preds_dess_volumes[patient][visit][start:end, ...]
    #         except (KeyError,) as e:
    #             print(f"{patient}:{visit}, ", end='')
    # print('')

    print(f"Calculating DESS volume metrics")
    for patient, visits in preds_dess_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.365 * 0.456 * 0.7)
            if patient not in preds_dess_totals.keys():
                preds_dess_totals[patient] = [v, ]
            else:
                preds_dess_totals[patient].append(v)

    ################ Missing Data Check One ########################################################
    missing_iwfs = preds_dess_totals.keys() - preds_iwfs_totals.keys()
    missing_dess = preds_iwfs_totals.keys() - preds_dess_totals.keys()
    iwfs_dess_intersection = preds_dess_totals.keys() & preds_iwfs_totals.keys()
    print(f"> {len(missing_iwfs)} cases missing in IWFS, {len(missing_dess)} cases missing in DESS.")
    print(f"> {len(iwfs_dess_intersection)} usable cases, so far.")
    print("Missing IWFS Data: (Patients in DESS not in IWFS)")
    for p in sorted(missing_iwfs):
        print(f"{p} ", end='')
    print('')
    print("Missing DESS Data: (Patients in IWFS not in DESS)")
    for p in sorted(missing_dess):
        print(f"{p} ", end='')
    print('')
    #################################################################################################

    print(f"Calculating IWFS - DESS")
    # NOTE only uses vol[0], so ignores multiple visits!
    diff_iwfs_dess = {}  # Key: Patient, Val: Diff of Volumes
    for patient, iwfs_volumes in preds_iwfs_totals.items():
        assert len(iwfs_volumes) > 0
        if patient in preds_dess_totals.keys():
            assert len(preds_dess_totals[patient]) > 0
            diff_iwfs_dess[patient] = abs(iwfs_volumes[0] - preds_dess_totals[patient][0])

    print(f"Loading BML from {manual_bml_masks}")
    # Find Manual BML Mask Volumes
    manual_bml_series = files.read_image_series(manual_bml_masks, ends_with="mask.bmp")
    manual_bml_series = common.dictutils.slices_to_mask(manual_bml_series, threshold=0.5)
    manual_bml_volumes = common.dictutils.visit_to_volume(manual_bml_series)
    manual_bml_totals = {}  # Key: Patient, Val: List of Volumes

    print(f"Calculating BML volume metrics")
    for patient, visits in manual_bml_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            if patient not in manual_bml_totals.keys():
                manual_bml_totals[patient] = [v, ]
            else:
                manual_bml_totals[patient].append(v)

    ################ Missing Data Check Two ########################################################
    missing_bml = diff_iwfs_dess.keys() - manual_bml_totals.keys()
    missing_preds = manual_bml_totals - diff_iwfs_dess.keys()
    bml_scans_intersection = diff_iwfs_dess.keys() & manual_bml_totals.keys()
    print(f"> {len(missing_bml)} patients in scans missing BML, {len(missing_preds)} cases in BML masks missing scan preds.")
    print(f"> {len(bml_scans_intersection) + len(missing_bml)} usable cases, total.")  # Note: Accounts for below
    print("Missing BML Data: (Patients in Preds Data without BML, Set to Zero)")
    for p in sorted(missing_bml):
        print(f"{p} ", end='')
        manual_bml_totals[p] = [0, ]  # Note: This sets patients without BML data to BML=0, assuming it's intentional.
    print('')
    print("Missing Bone Preds: (Patients in BML masks without Bone predictions present in both DESS *and* IWFS)")
    for p in sorted(missing_preds):
        print(f"{p} ", end='')
    print('')
    #################################################################################################

    print(f"Finding Pearson Correlation of (IWFS - DESS) to BML.")
    pearson_intersection = diff_iwfs_dess.keys() & manual_bml_totals.keys()
    pearson_x = []
    pearson_y = []
    for patient in pearson_intersection:
        pearson_x.append(manual_bml_totals[patient][0])
        pearson_y.append(diff_iwfs_dess[patient])

    pearson_x = np.array(pearson_x)
    pearson_y = np.array(pearson_y)
    r, p_val = scipy.stats.pearsonr(pearson_x, pearson_y)
    print(f"Pearson R: {r}\nP-Value: {p_val}")

    plt.scatter(pearson_x, pearson_y, alpha=0.7)
    plt.title("BML vs (IWFS - DESS) Volume Measurement")
    plt.xlabel("BML Segmentation Volume")
    plt.ylabel("(IWFS - DESS) Volume")
    plt.figtext(0.99, 0.01, f"r = {r : .4f}", horizontalalignment='right')
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
