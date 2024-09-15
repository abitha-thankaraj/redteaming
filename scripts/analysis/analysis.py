import pandas as pd
from redteam.utils.data_utils import read_json, write_json
import numpy as np


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


if __name__ == "__main__":
    df = read_and_parse_csv(
        "/data/tir/projects/tir7/user_data/athankar/redteaming/Redteaming - evals.csv"
    )
    df["REWARD_RELABELLED_FNAME"] = df["FNAME"].str.replace(".json", "_llama_rewards.json")
    # Initialize new columns
df["num_evals"] = 0
df["num_jailbreaks"] = 0
df["num_jailbreaks_per_turn"] = None  # This will be a list for each row

for index, row in df.iterrows():
    reward_relabelled_fname = row["REWARD_RELABELLED_FNAME"]
    reward_matrix = np.array(pd.read_json(reward_relabelled_fname)["rewards"].to_list())

    # Calculate metrics
    num_evals = reward_matrix.shape[0]
    num_jailbreaks_per_turn = reward_matrix.sum(
        axis=0
    ).tolist()  # Convert to list for storage in DataFrame
    num_jailbreaks = sum(reward_matrix.sum(axis=1) > 0.0)

    # Assign values to the DataFrame
    df.at[index, "num_evals"] = num_evals
    df.at[index, "num_jailbreaks"] = num_jailbreaks
    df.at[index, "num_jailbreaks_per_turn"] = num_jailbreaks_per_turn

df["model"] = df["Attacker-model"].apply(
    lambda x: "mistral" if "mistral" in str(x).lower() else "llama"
)
# df.to_pickle("/data/tir/projects/tir7/user_data/athankar/redteaming/analysis.pkl")  # where to save it, usually as a .pkl
