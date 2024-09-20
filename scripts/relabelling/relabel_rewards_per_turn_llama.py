import argparse
from tqdm import tqdm
from redteam.utils.data_utils import read_json, write_json
from redteam.inference.judge import LlamaGuardJudge
from redteam.utils.slack_me import slack_notification


def main(fname: str, device: str):
    save_fname = fname.replace(".json", "_llama_rewards.json")
    judge = LlamaGuardJudge(device=device)
    dataset = read_json(fname)
    for i, data in tqdm(enumerate(dataset)):
        conversation = data["conversation"]
        data.update(judge.score(conversation))
        if i % 20 == 0:
            write_json(dataset, save_fname)

    write_json(dataset, save_fname)
    slack_notification(f"Finished processing {fname} and saved to {save_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        type=str,
        default="/data/group_data/rl/datasets/redteaming/gen_judge_multiturn_conversation_combined/combined_eval_data.json",
        help="Path to the json file",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()
    main(args.fname, args.device)
