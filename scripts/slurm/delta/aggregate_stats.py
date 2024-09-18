import os, glob, yaml
import pandas as pd
from redteam.utils.data_utils import read_json
import numpy as np

EVAL_DIR = '/scratch/bcgv/datasets/redteaming/redteaming_evals'

def find_traj_file_dataset(fnames):
    traj_file = None
    dataset = None
    for fname in fnames:
        if "openai" in fname:
            traj_file = fname
            dataset = "openai"
            break
        elif "jailbreakbench" in fname:
            traj_file = fname
            dataset = "jailbreakbench"
            break
    return traj_file, dataset

def aggregate_results(trajs):
    rewards = []
    for traj in trajs:
        rewards.append(traj["judge"]["rewards"])
    rewards = np.array(rewards)
    # TODO: Value function metrics
    return {
        "jailbreaks_per_turn": np.sum(rewards, axis=0).tolist(),
        "num_jailbreaks": int(np.sum(np.any(rewards == 1., axis=1))),
        "num_evals": len(trajs)
    }
if __name__ == "__main__":
    eval_folders = glob.glob(os.path.join(EVAL_DIR, '**'))
    eval_folders = [e for e in eval_folders if os.path.isdir(e) and ("rwr" in e or "sft" in e or "untrained" in e)]
    reprocess_list = []
    df = pd.DataFrame(columns=['attacker_training_method', 'attacker_checkpoint', 'defender_training_method', 
                               'defender_checkpoint', 'defender_lr', 'dataset', 'traj_file', 'results_file', 
                               'temperature', 'num_jailbreaks', 'num_jailbreaks_per_turn', 'num_evals', 'config', 'flagged'])
    
    for folder in eval_folders:
        defender_lr = folder.split("_")[-1]
        defender_training_method = "untrained" if defender_lr in ["attacker"] else None

        subfolders = glob.glob(os.path.join(folder, '**/**/'))
        for subfolder in subfolders:
            fnames = glob.glob(os.path.join(subfolder, '*'))
            if len(fnames) > 3:
                reprocess_list.append((subfolder, defender_lr))
                continue

            cfg_file = os.path.join(subfolder, 'config.yaml')
            with open(cfg_file, 'r') as f:
                config = yaml.safe_load(f)
            
            results_file = os.path.join(subfolder, '_results.json')
            if not os.path.exists(results_file):
                continue

            results = pd.read_json(results_file)


            traj_file, dataset = find_traj_file_dataset(fnames)

            if traj_file and dataset:
                new_row = pd.DataFrame({
                    'attacker_training_method': [config["attacker"]["model_type"]],
                    'attacker_checkpoint': [config["attacker"]["model_dir"]],
                    'defender_training_method': [config["defender"]["model_type"]],
                    'defender_checkpoint': [config["defender"]["model_dir"]],
                    'defender_lr': [defender_lr],
                    'dataset': [dataset],
                    'traj_file': [traj_file],
                    'results_file': [results_file],
                    'temperature': [config["defender"]["generation_kwargs"]["temperature"]],
                    'num_jailbreaks': [results["num_jailbreaks"].iloc[0]],
                    'num_jailbreaks_per_turn': [np.array(results["jailbreaks_per_turn"])],
                    'config': [config],
                    'num_evals': [results["num_evals"].iloc[0]],
                    'flagged': False
                })
                df = pd.concat([df, new_row], ignore_index=True)

    
    # print(reprocess_list)
    for folder, defender_lr in reprocess_list:
        fnames = glob.glob(os.path.join(folder, '*'))
        traj_files = [f for f in fnames if "openai" in f or "jailbreakbench" in f]
        cfg_file = os.path.join(subfolder, 'config.yaml')
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        temperature = config["defender"]["generation_kwargs"]["temperature"]

        for traj_file in traj_files:
            dataset = "openai" if "openai" in traj_file else "jailbreakbench"
            attacker_training_method = "rwr_trained_attacker" if "rwr_trained_attacker" in traj_file else "sft_trained_attacker"
            defender_training_method = "rwr_trained_defender" if "rwr_trained_defender" in traj_file else "sft_trained_defender"

            trajs = read_json(traj_file)
            results = aggregate_results(trajs)
        
            new_row = pd.DataFrame({
                'attacker_training_method': [attacker_training_method],
                'attacker_checkpoint': None,
                'defender_training_method': [defender_training_method],
                'defender_checkpoint': None,
                'defender_lr': [defender_lr],
                'dataset': [dataset],
                'traj_file': [traj_file],
                'results_file': None,
                'temperature': [temperature],
                'num_jailbreaks': [results["num_jailbreaks"]],
                'num_jailbreaks_per_turn': [results["jailbreaks_per_turn"]],
                'config': [config],
                'num_evals': [results["num_evals"]],
                'flagged': True
            })
            df = pd.concat([df, new_row], ignore_index=True)
    
    from IPython import embed; embed()

# df.sort_values(["dataset","defender_training_method", "defender_lr", "attacker_training_method", "temperature"], ascending=[True, True, False, True, True]).to_csv("agg.csv", index=False)  