import argparse
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    args = parser.parse_args()


    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)
    