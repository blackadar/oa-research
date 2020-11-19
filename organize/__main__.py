"""
Command-Line Interface for Organizing the OA Datasets.
This can generate "meta" folders with masks and images in formats for ML algorithms.
It will also help get data out of this structure, into "train", "test", and "validate" as desired.
"""
from organize import make_meta, split_meta
from os import system, name
import pathlib


def main():
    op = list_prompt("Choose an Operation", ["Generate Meta Folders", "Create Data from Meta Folders", "Exit"])

    if op == 0:
        # Generate meta folders
        path = path_prompt("Enter Patient Folder", check_exists=True)
        options = multi_list_prompt("Select Data to Generate", ["Images", "BML Masks", "Bone Masks",
                                                                "Bone Segmented Images"])
        images = 0 in options
        bml = 1 in options
        bone = 2 in options
        bone_segmented = 3 in options
        make_meta.make_meta_for_patients(path, images=images, bml=bml, bone=bone, bone_segmented_images=bone_segmented)
        done()
    elif op == 1:
        # Create Data from Meta Folders
        num_phase = int_prompt("How many Phase folders?")
        phases = []
        weights = []
        for p in range(0, num_phase):
            phases.append(path_prompt(f"Enter Phase Path {p+1}."))
            train = float_prompt(f"Enter amount of this data allocated to training, 0.0 - 1.0.")
            test = float_prompt(f"Enter amount of this data allocated to testing, 0.0 - 1.0.")
            validate = float_prompt(f"Enter amount of this data allocated to validation, 0.0 - 1.0.")
            weights.append((train, test, validate))

        dat = list_prompt("Choose the data configuration.", ["Bone Segmentation (images, bone masks)",
                                                             "Raw BML (images, bml masks)",
                                                             "Bone Segmented BML (bone segmented images, bml masks)"])
        output = path_prompt("Enter output directory for train, test, validate folders.", check_exists=False)

        patients = split_meta.split_phases(phases, weights)
        if dat == 0:
            split_meta.bone_segmentation(output, patients, only_v00=True)
        if dat == 1:
            split_meta.raw_bml(output, patients, only_v00=True)
        if dat == 2:
            split_meta.bone_segmented_bml(output, patients, only_v00=True)
        done()
    elif op == 2:
        # Exit
        exit(0)


def list_prompt(prompt, options, context=None):
    clear()
    print(f"> {prompt} <")
    for idx, option in enumerate(options):
        print(f" {idx+1}. {option}")
    print()
    if context:
        print(f"* {context} *")
    selection = input("> ")

    # Try to parse user input
    try:
        selection = int(selection.strip())
    except ValueError:
        list_prompt(prompt, options, context=f"'{selection}' is not an integer. Try again.")
    selection = selection - 1  # User entered one-based index
    if selection not in range(0, len(options)):
        list_prompt(prompt, options, context=f"'{selection + 1}' is not a selection. Try again.")

    clear()
    return selection


def multi_list_prompt(prompt, options, context=None):
    clear()
    print(f"> {prompt} <")
    for idx, option in enumerate(options):
        print(f" {idx + 1}. {option}")
    print()
    if context:
        print(f"* {context} *")
    print("Enter multiple selections separated by commas. (e.g. 1, 2, 3)")
    inp = input("> ")

    selections = inp.split(",")
    cleaned = []

    # Try to parse user input
    for s in selections:
        try:
            i = int(s.strip())
        except ValueError:
            multi_list_prompt(prompt, options, context=f"'{s}' is not an integer. Try again.")
            break
        i -= 1  # User entered one-based index
        if i not in range(0, len(options)):
            multi_list_prompt(prompt, options, context=f"'{i + 1}' is not a selection. Try again.")
        cleaned.append(i)

    if len(cleaned) < 1:
        multi_list_prompt(prompt, options, context=f"Please make a selection.")

    clear()
    return cleaned


def path_prompt(prompt, check_exists=True, context=None):
    clear()
    print(f"> {prompt} <")
    if context:
        print(f"* {context} *")
    path = input("> ")

    # Try to parse user input
    try:
        path = pathlib.Path(path)
    except Exception as e:
        path_prompt(prompt, check_exists=check_exists, context=f"Invalid Path: {e}. Try again.")

    if check_exists:
        if not path.exists():
            path_prompt(prompt, check_exists=check_exists, context=f"Path '{path}' does not exist. Try again.")

    clear()
    return path


def int_prompt(prompt, context=None):
    clear()
    print(f"> {prompt} <")
    print()
    if context:
        print(f"* {context} *")
    x = input("> ")

    # Try to parse user input
    try:
        i = int(x.strip())
    except ValueError:
        return int_prompt(prompt, context=f"'{x}' is not an integer. Try again.")

    clear()
    return i


def float_prompt(prompt, context=None):
    clear()
    print(f"> {prompt} <")
    print()
    if context:
        print(f"* {context} *")
    x = input("> ")

    # Try to parse user input
    try:
        i = float(x.strip())
    except ValueError:
        return float_prompt(prompt, context=f"'{x}' is not a float. Try again.")

    clear()
    return i


def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')

        # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')


def done():
    print("Complete.")


if __name__ == "__main__":
    main()
