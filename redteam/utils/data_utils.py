import json
import os
from typing import Optional, List


def read_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def write_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_all_files(
    root_dir: str, 
    file_suffix: Optional[str] = None,
) -> List[str]:
    """
    Given the root directory, returns the absolute path to all files.
    Filters by suffix if given one.

    Input:
        root_dir (str): The absolute path to the root directory where all files should be in.
        file_suffix (Optional[str]): The suffix (e.g., json) to filter by. 
            Default: None
    
    Output:
        A List of str, with absolute paths to all files in the directory with the given suffix.

    Example Usage:
        all_data_files = find_all_files(
            root_dir='/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_chunked',
            file_suffix='json',
        )

        for data_file in all_data_files:
            pass
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Given directory {root_dir} is not a directory.")
    
    all_sub_file_or_dirs = os.listdir(root_dir)

    all_files = []
    for sub_file_or_dir in all_sub_file_or_dirs:
        if os.path.isfile(sub_file_or_dir):
            if file_suffix is None or sub_file_or_dir.endswith(file_suffix):
                absolute_path = os.path.join(root_dir, sub_file_or_dir)
                all_files.append(absolute_path)

    return all_files
