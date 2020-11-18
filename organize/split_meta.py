"""
Splits meta sub-directories into train, test, and validate.
Functions for each model, as well as a general function that you can modify for new meta folders if you'd like.
See make_meta.py for how these directories were made.
"""
import math
import pathlib
import shutil
import json
import datetime
import common.dictutils as du


def split_patients(path, train=0.7, test=0.15, validate=0.15):
    """
    Divide the data into train, test, and validate on the patient level.
    This prevents unfair bias from preview into the test/validate sets.
    :param path: path-like to the directory containing patient directories
    :param train: float 0.0 - 1.0 amount of total data to allocate to training
    :param test: float 0.0 - 1.0 amount of total data to allocate to testing
    :param validate: float 0.0 - 1.0 amount of total data to allocate to validation
    :return: dict {'train': [...], 'test': [...], 'validate': [...]}
    """
    assert train + test + validate == 1.0, "Train, Test, and Validate must add to 1.0."
    result = {
            'train': [],
            'test': [],
            'validate': [],
    }

    folder = pathlib.Path(path)
    patients = [patient for patient in folder.iterdir() if patient.is_dir()]
    num_patients = len(patients)
    num_train = math.floor(num_patients * train)
    num_test = math.floor(num_patients * test)
    num_validate = num_patients - (num_train + num_test)

    for i in range(0, num_train):
        result['train'].append(patients.pop(0))
    for i in range(0, num_test):
        result['test'].append(patients.pop(0))
    for i in range(0, num_validate):
        result['validate'].append(patients.pop(0))

    print(f"Actual Patient Split is train: {num_train} ({num_train/num_patients: 0.2f}) test: {num_test} "
          f"({num_test/num_patients: 0.2f}) "
          f"validate: {num_validate} ({num_validate/num_patients: 0.2f}) ")

    return result


def split_phases(paths, weights):
    """
    Splits PhaseX folders and combines their output into one patient split.
    Caller specifies paths as a list and weights as a list of (train, test, validate) tuples, equal length.
    :param paths: List of path-likes to folders containing patients
    :param weights: List of (train, test, validate) tuples (each is a 0.0 - 1.0 float), equal in length to paths
    :return: dict {'train': [...], 'test': [...], 'validate': [...]}
    """
    assert len(paths) == len(weights), "Specify an equal set of weights for each path."
    result = {
            'train':    [],
            'test':     [],
            'validate': [],
    }

    for path, (train, test, validate) in zip(paths, weights):
        r = split_patients(path, train, test, validate)
        result['train'].extend(r['train'])
        result['test'].extend(r['test'])
        result['validate'].extend(r['validate'])

    return result


def _build_general(baseline_meta, target_meta, out_dir, patients_split, only_v00=True):
    """
    Builds a general model dataset, from meta subdirectories and outputs split patients.
    :param baseline_meta: str, Directory in meta, containing model's baseline images
    :param target_meta: str, Directory in meta, containing model's target (mask)
    :param out_dir: path-like Directory to place train, test, validate folders
    :param patients_split: Dict {'train': [...], 'test': [...], 'validate': [...]} (use split_patients)
    :param only_v00: Only copy data from v00, the first visit
    :return: None
    """
    out_dir = pathlib.Path(out_dir)
    train_dir = out_dir/"train"
    test_dir = out_dir/"test"
    val_dir = out_dir/"validate"
    out_dir.mkdir(exist_ok=True, parents=True)
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)
    tracker = {
            'info': {
                    'baseline_meta': str(baseline_meta),
                    'target_meta': str(target_meta),
                    'output_dir': str(out_dir),
                    'only_v00': only_v00,
                    'timestamp': str(datetime.datetime.now())
            },
            'train': {},
            'test': {},
            'validate': {},
    }

    def work(iterable, out_dir, tracker):
        for p in iterable:
            tracker[p.name] = {}
            visits = [p/'v00', ] if only_v00 else [visit for visit in p.iterdir() if visit.is_dir()]
            for v in visits:
                tracker[p.name][v.name] = {
                        'baselines': [],
                        'targets': [],
                }
                meta = v/'meta'
                b_meta = meta/baseline_meta
                t_meta = meta/target_meta
                assert meta.exists(), f"{v} does not contain a meta directory."
                assert b_meta.exists(), f"{v} does not contain a baseline '{baseline_meta}' meta directory."
                assert t_meta.exists(), f"{v} does not contain a target '{target_meta}' meta directory."
                t_imgs = []
                t_imgs_corr = []
                b_imgs = []
                for t_img in t_meta.iterdir():
                    if t_img.is_file():
                        assert '_mask' in t_img.name, "Target must have 'mask' in the name, otherwise the files cannot be associated with the baseline."
                        shutil.copy(t_img, out_dir/t_img.name)
                        t_imgs.append(t_img.name)
                        t_imgs_corr.append(t_img.name.replace('_mask', ''))
                for b_img in b_meta.iterdir():
                    if b_img.is_file() and b_img.name in t_imgs_corr:
                        shutil.copy(b_img, out_dir/b_img.name)
                        b_imgs.append(b_img.name)
                tracker[p.name][v.name]['baselines'] = b_imgs
                tracker[p.name][v.name]['targets'] = t_imgs

    work(patients_split['train'], train_dir, tracker['train'])
    work(patients_split['test'], test_dir, tracker['test'])
    work(patients_split['validate'], val_dir, tracker['validate'])

    json_tracker = out_dir/'contents.json'
    with json_tracker.open('w') as js:
        js.write(json.dumps(tracker, indent=4))


def bone_segmentation(out_dir, patients_split, only_v00=True):
    """
    Copies data for the Bone Segmentation model into the desired out_dir.
    :param out_dir: path-like Directory to place train, test, validate folders
    :param patients_split: Dict {'train': [...], 'test': [...], 'validate': [...]} (use split_patients)
    :param only_v00: Only copy data from v00, the first visit
    :return: None
    """
    _build_general('images', 'bone_masks', out_dir, patients_split, only_v00)


def raw_bml(out_dir, patients_split, only_v00=True):
    """
    Copies data for the raw image BML Segmentation model into the desired out_dir.
    :param out_dir: path-like Directory to place train, test, validate folders
    :param patients_split: Dict {'train': [...], 'test': [...], 'validate': [...]} (use split_patients)
    :param only_v00: Only copy data from v00, the first visit
    :return: None
    """
    _build_general('images', 'bml_masks', out_dir, patients_split, only_v00)


def bone_segmented_bml(out_dir, patients_split, only_v00=True):
    """
    Copies data for the model which predicts BML from manually segmented bone into the desired out_dir.
    :param out_dir: path-like Directory to place train, test, validate folders
    :param patients_split: Dict {'train': [...], 'test': [...], 'validate': [...]} (use split_patients)
    :param only_v00: Only copy data from v00, the first visit
    :return: None
    """
    _build_general('bone_segmented_images', 'bml_masks', out_dir, patients_split, only_v00)
