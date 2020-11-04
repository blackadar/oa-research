"""
Finds the volume measurement from the preds output of a model.
"""
import pathlib
import common.files as files
import common.imutils as iu
import analysis.volume as vol

preds = pathlib.Path('data/iwfs_out/')


def main():
    preds_series = files.read_image_series(preds, ends_with="pred.png")
    preds_series = iu.slices_to_mask(preds_series, threshold=0.5)
    preds_volumes = iu.visit_to_volume(preds_series)

    print("Volumetric Analysis")

    for patient, visits in preds_volumes.items():
        for visit, volume in visits.items():
            v = vol.mask(volume, voxel_size=0.357 * 0.511 * 3)
            print(f"{patient}, {visit}: {v: .3f} mm^3.")

    print("Per-Slice Analysis")

    for patient, visits in preds_series.items():
        for visit, slice_ids in visits.items():
            print(f"== {patient} {visit} ==")
            for slice_id, im in slice_ids.items():
                v = vol.mask(im, voxel_size=0.357 * 0.511 * 3)
                print(f"  {slice_id}: {v: .3f} mm^3")


if __name__ == "__main__":
    main()
