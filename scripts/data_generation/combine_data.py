import os
import json
import glob


def combine_json_files(folder_pattern, json_pattern="*.json", out_file_pattern=""):
    combined_data = []

    # Find folders matching the pattern
    matching_folders = glob.glob(folder_pattern)

    for folder in matching_folders:
        # Find JSON files in the folder
        json_files = glob.glob(os.path.join(folder, json_pattern))

        for json_file in json_files:
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    combined_data.extend(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_file}")

    from IPython import embed

    embed()
    SAVE_FOLDER = "/scratch/bcgv/datasets/redteaming/gen_judge_multiturn_conversation_combined/iterative/"
    # Write combined data to a new JSON file
    with open(f"{SAVE_FOLDER}combined_{out_file_pattern}.json", "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Combined data from {len(combined_data)} JSON files")


if __name__ == "__main__":
    FOLDER = "/scratch/bcgv/datasets/redteaming/redteaming_evals/"

    patterns = [
        "untrained_defender.sft_trained_attacker.iter0",
        "rwr_trained_attacker.untrained_defender.iter0",
        "sft_trained_attacker.sft_defender.iter1",
        "rwr_trained_attacker.rwr_defender.iter1",
    ]

    for pattern in patterns:
        folder_pattern = f"{FOLDER}{pattern}**/**/**"
        combine_json_files(folder_pattern, "**ultrasafety**.json", out_file_pattern=pattern)
