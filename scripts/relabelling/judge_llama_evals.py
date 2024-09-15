import argparse
import copy
from tqdm import tqdm
from redteam.utils.data_utils import read_json, write_json
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.slack_me import slack_notification
from redteam.utils.timer import DebugTimer

import pandas as pd


def read_and_parse_csv(file_path):
    """
    Reads a CSV file and parses it into a pandas DataFrame.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The parsed DataFrame.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, skipinitialspace=True)

        # Drop rows where all columns are NaN (empty rows)
        df = df.dropna(how="all")

        # Reset the index after dropping rows
        df = df.reset_index(drop=True)

        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(
            f"Error: Unable to parse the file '{file_path}'. Please ensure it's a valid CSV."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


def main(fname: str, judge: LlamaGuardJudge):

    save_fname = fname.replace(".json", "_llama_rewards.json")
    dataset = read_json(fname)
    for i, data in tqdm(enumerate(dataset)):
        conversation = copy.deepcopy(data["game"][1:])
        for turn in range(len(conversation)):
            if conversation[turn]["role"] == "attacker":
                conversation[turn]["role"] = "user"
            elif conversation[turn]["role"] == "defender":
                conversation[turn]["role"] = "assistant"

        dataset[i].update(judge.score(conversation))

    write_json(dataset, save_fname)


if __name__ == "__main__":
    df = read_and_parse_csv(
        "/data/tir/projects/tir7/user_data/athankar/redteaming/Redteaming - evals.csv"
    )
    fnames = df["FNAME"]
    judge = LlamaGuardJudge(device="cuda:0")
    # main(fname=fnames[19], judge=judge)
    for fname in fnames:
        with DebugTimer(f"Processing {fname}") as timer:
            main(fname=fname, judge=judge)
            slack_notification(
                f"Finished processing {fname} - elapsed_time = {timer.get_time_str()}"
            )
