from tqdm import tqdm

from redteam.utils.data_utils import read_json, write_json
from redteam.utils.slack_me import slack_notification


def get_labels(rewards):
    assert len(rewards) == 3, rewards
    labels = []
    for i, reward in enumerate(rewards):
        if reward == 0.0:
            labels.append("<UNSAFE> ")
        elif reward == 1.0:
            if i == len(rewards) - 1:
                labels.append("<SAFE> ")
            elif rewards[i + 1] == 1.0:
                labels.append("<SAFE> ")
            else:
                labels.append("<MODERATE> ")

    return labels


LABELS = {
    (1.0, 1.0, 1.0): ["<SAFE> ", "<SAFE> ", "<SAFE> "],
    (1.0, 1.0, 0.0): ["<SAFE> ", "<SAFE> ", "<UNSAFE> "],
    (1.0, 0.0, 1.0): ["<MODERATE> ", "<UNSAFE> ", "<SAFE> "],
    (1.0, 0.0, 0.0): ["<MODERATE> ", "<UNSAFE> ", "<UNSAFE> "],
    (0.0, 1.0, 1.0): ["<UNSAFE> ", "<SAFE> ", "<SAFE> "],
    (0.0, 1.0, 0.0): ["<UNSAFE> ", "<MODERATE> ", "<UNSAFE> "],
    (0.0, 0.0, 1.0): ["<UNSAFE> ", "<UNSAFE> ", "<SAFE> "],
    (0.0, 0.0, 0.0): ["<UNSAFE> ", "<UNSAFE> ", "<UNSAFE> "],
}

REPAIRED_LABELS = {
    (1.0, 1.0, 1.0): ["<SAFE> ", "<SAFE> ", "<SAFE> "],
    (1.0, 1.0, 0.0): ["<SAFE> ", "<SAFE> ", "<UNSAFE> <FIX>  "],
    (1.0, 0.0, 1.0): ["<MODERATE> ", "<UNSAFE> <FIX>  ", "<SAFE> "],
    (1.0, 0.0, 0.0): ["<MODERATE> ", "<UNSAFE> <FIX>  ", "<UNSAFE> <FIX>  "],
    (0.0, 1.0, 1.0): ["<UNSAFE> <FIX>  ", "<SAFE> ", "<SAFE> "],
    (0.0, 1.0, 0.0): ["<UNSAFE> <FIX>  ", "<MODERATE> ", "<UNSAFE> <FIX>  "],
    (0.0, 0.0, 1.0): ["<UNSAFE> <FIX>  ", "<UNSAFE> <FIX>  ", "<SAFE> "],
    (0.0, 0.0, 0.0): ["<UNSAFE> <FIX>  ", "<UNSAFE> <FIX>  ", "<UNSAFE> <FIX>  "],
}


def label_conversation(conversation, rewards, is_repaired=False):

    if is_repaired:
        labels = REPAIRED_LABELS[tuple(rewards)]
    else:
        labels = LABELS[tuple(rewards)]
    for i in range(len(conversation)):
        if i % 2 == 1 and conversation[i]["role"] == "assistant":
            conversation[i]["content"] = labels[i // 2] + conversation[i]["content"]

    return conversation


def save_labeled_conversations(fname):
    # read the file
    records = read_json(fname=fname)
    # label the conversations

    new_records = []

    for record in tqdm(records):

        new_record = {}
        rewards = record["chosen_reward"]
        conversation = record["chosen_conversation"]
        labeled_conversation = label_conversation(conversation, rewards)
        new_record["chosen_conversation"] = labeled_conversation
        new_record["chosen_reward"] = rewards

        rejected_reward = record["rejected_reward"]
        rejected_conversation = record["rejected_conversation"]
        rejected_labeled_conversation = label_conversation(
            rejected_conversation, rejected_reward
        )
        new_record["rejected_conversation"] = rejected_labeled_conversation
        new_record["rejected_reward"] = rejected_reward

        new_records.append(new_record)

        new_record = {}
        labeled_repaired_conversation = label_conversation(
            record["repaired_conversation"], record["rejected_reward"], is_repaired=True
        )
        new_record["chosen_conversation"] = labeled_repaired_conversation
        new_record["chosen_reward"] = record["rejected_reward"]
        new_record["rejected_conversation"] = rejected_labeled_conversation
        new_record["rejected_reward"] = rejected_reward

        new_records.append(new_record)

    # save the labeled conversations
    out_fname = fname.replace(".json", "_value_labeled.json")
    write_json(new_records, out_fname)
    slack_notification(f"Saved labeled conversations to {out_fname}")


def save_conversations(fname):
    # read the file
    records = read_json(fname=fname)
    # label the conversations

    new_records = []

    for record in tqdm(records):

        new_record = {}
        rewards = record["chosen_reward"]
        conversation = record["chosen_conversation"]
        new_record["chosen_conversation"] = conversation
        new_record["chosen_reward"] = rewards

        rejected_reward = record["rejected_reward"]
        rejected_conversation = record["rejected_conversation"]
        new_record["rejected_conversation"] = rejected_conversation
        new_record["rejected_reward"] = rejected_reward

        new_records.append(new_record)

        new_record = {}
        repaired_conversation = record["repaired_conversation"]
        # label_conversation(record["repaired_conversation"], record["rejected_reward"], is_repaired=True)
        new_record["chosen_conversation"] = repaired_conversation
        new_record["chosen_reward"] = record["rejected_reward"]
        new_record["rejected_conversation"] = rejected_conversation
        new_record["rejected_reward"] = rejected_reward
        new_records.append(new_record)

    # save the labeled conversations
    out_fname = fname.replace(".json", "_unlabeled.json")
    write_json(new_records, out_fname)
    slack_notification(f"Saved labeled conversations to {out_fname}")


if __name__ == "__main__":

    fnames = [
        "/data/group_data/rl/datasets/redteaming/best_of_n_high_contrast_1000/2024.09.26/00-12-1727323973/best_of_n_repaired_conversations_best_of_n_high_contrast_1000.json",
        "/data/group_data/rl/datasets/redteaming/hard_negatives_1000_continued/2024.09.26/08-14-1727352850/best_of_n_repaired_conversations_hard_negatives_1000_continued.json",
        "/data/group_data/rl/datasets/redteaming/hard_negatives_1000/2024.09.26/00-19-1727324358/best_of_n_repaired_conversations_hard_negatives_1000.json",
    ]

    for fname in fnames:
        save_labeled_conversations(fname)
        save_conversations(fname)
