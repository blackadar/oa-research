"""
Command-Line Interface for Organizing the OA Datasets.
This can generate "meta" folders with masks and images in formats for ML algorithms.
It will also help get data out of this structure, into "train", "test", and "validate" as desired.
"""
from organize import make_meta, split_meta
from os import system, name, getcwd
import pathlib
import sys


def main():
    op = list_prompt("Choose an Operation", ["Generate Meta Folders", "Create Data from Meta Folders", "Exit"])

    if op == 0:
        # Generate meta folders
        path = path_prompt(f"Enter Patient Folder (absolute or relative to {getcwd()})", check_exists=True)
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
        use_first = []
        weights = []
        for p in range(0, num_phase):
            phases.append(path_prompt(f"Enter Phase Path {p+1}. (absolute or relative to {getcwd()})"))
            partial = list_prompt(f"Would you like to use a partial amount of patients in {phases[p]}?",
                                  ["Yes, specify the number of patients to use.", "No, use all patients in folder."])
            if partial == 0:
                use_first.append(int_prompt(f"How many patients (first n) should be used from {phases[p]}?"))
            else:
                use_first.append(None)  # Signals to use all
            train = float_prompt(f"Enter percentage of {phases[p]} patients allocated to training, 0.0 - 1.0.")
            test = float_prompt(f"Enter percentage of {phases[p]} patients allocated to testing, 0.0 - 1.0.")
            validate = float_prompt(f"Enter percentage of {phases[p]} patients allocated to validation, 0.0 - 1.0.")
            weights.append((train, test, validate))

        dat = list_prompt("Choose the data configuration.", ["Bone Segmentation (images, bone masks)",
                                                             "Raw BML (images, bml masks)",
                                                             "Bone Segmented BML (bone segmented images, bml masks)"])
        output = path_prompt("Enter output directory for train, test, validate folders.", check_exists=False)

        # Check Output is Writeable
        try:
            output.resolve()
        except OSError as e:
            print(f"Invalid path '{output}': {e}")
            sys.exit(-1)

        # Check Output is on a pre-approved drive
        if output.resolve().drive not in ("F:", "C:", "f:", "c:"):
            confirm = list_prompt(f"{output} seems like an irregular choice. Are you sure? "
                                  f"You probably want to write to the Student drive.",
                                  [f"Yes, output to {output}.", "No, cancel operation."])
            if confirm == 1:
                print("Please re-run the script to start your operation correctly.")
                sys.exit(0)

        patients = split_meta.split_phases(phases, weights, use_first)
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
        return list_prompt(prompt, options, context=f"'{selection}' is not an integer. Try again.")
    selection = selection - 1  # User entered one-based index
    if selection not in range(0, len(options)):
        return list_prompt(prompt, options, context=f"'{selection + 1}' is not a selection. Try again.")

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
            return multi_list_prompt(prompt, options, context=f"'{s}' is not an integer. Try again.")
        i -= 1  # User entered one-based index
        if i not in range(0, len(options)):
            return multi_list_prompt(prompt, options, context=f"'{i + 1}' is not a selection. Try again.")
        cleaned.append(i)

    if len(cleaned) < 1:
        return multi_list_prompt(prompt, options, context=f"Please make a selection.")

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
        return path_prompt(prompt, check_exists=check_exists, context=f"Invalid Path: {e}. Try again.")

    if check_exists:
        if not path.exists():
            return path_prompt(prompt, check_exists=check_exists, context=f"Path '{path}' does not exist. Try again.")

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
