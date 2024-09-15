import os
import glob
import json

# from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from tqdm import tqdm
from redteam.utils.data_utils import read_json, write_json


if __name__ == "__main__":
    DATA_DIR = (
        "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_chunked"
    )

    train_data_jsons = glob.glob(DATA_DIR + "/**advbench**.json")
    train_data_jsons.extend(glob.glob(DATA_DIR + "/**harmbench**.json"))
    print(len(train_data_jsons))

    num_processes = 24

    with Pool(num_processes) as pool:
        data_list = list(
            tqdm(pool.imap(read_json, train_data_jsons), total=len(train_data_jsons))
        )
    combined_train_data = []
    for data in data_list:
        combined_train_data.extend(data)

    SAVE_DIR = (
        "/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined"
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(len(combined_train_data))
    write_json(combined_train_data, f"{SAVE_DIR}/combined_train_data.json")

    eval_data_jsons = glob.glob(DATA_DIR + "/**openai**.json")
    num_processes = 24
    with Pool(num_processes) as pool:
        eval_data_list = list(
            tqdm(pool.imap(read_json, eval_data_jsons), total=len(eval_data_jsons))
        )
    combined_eval_data = []
    for data in eval_data_list:
        combined_eval_data.extend(data)

    write_json(combined_eval_data, f"{SAVE_DIR}/combined_eval_data.json")
