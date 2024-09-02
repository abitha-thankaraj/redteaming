import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import os
import subprocess
import time
import json
import logging
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

base_controller_shell = '''python3 -m fastchat.serve.controller \\
    --host 0.0.0.0 \\
    --port {controller_port} > "{logdir}/controller.out" 2>&1'''

base_modelworker_shell = '''CUDA_VISIBLE_DEVICES={gpu_id} python3 -m fastchat.serve.model_worker \\
    --model-path {model_path} \\
    --model-name {model_name} \\
    --host 0.0.0.0 \\
    --controller-address "http://localhost:{controller_port}" \\
    --port {worker_port} \\
    --worker-address "http://localhost:{worker_port}" \\
    --device {device} > "{logdir}/{out_fname}.out" 2>&1'''

base_openai_shell = '''python3 -m fastchat.serve.openai_api_server \\
    --host 0.0.0.0 \\
    --port {openai_port} \\
    --controller-address "http://localhost:{controller_port}" > "{logdir}/openai_api_server.out" 2>&1'''

def get_conda_activate_cmd(cfg):
    return f"source {cfg.env.conda_profile} && conda activate {cfg.env.conda_env}"

def run_background(cfg, cmd, env):
    full_cmd = f"{get_conda_activate_cmd(cfg)} && {cmd}"
    return subprocess.Popen(full_cmd, shell=True, executable='/bin/bash', env=env)

def check_models_loaded(cfg):
    url = f"http://localhost:{cfg.ports.controller}/list_models"

    for attempt in range(cfg.max_load_retries):
        try:
            response = requests.post(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            models = response.json()["models"]
            if all(lambda model=model: model.name in models for model in cfg.models):
                logger.info("All models are loaded.")
                return True
            else:
                logger.warning("Not all models are loaded yet.")
        except RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
        
        # Exponential backoff
        time.sleep(2 ** attempt)
    
    logger.error(f"Failed to confirm all models loaded after {cfg.max_load_retries} attempts")
    return False

@hydra.main(version_base=None, config_path="/data/tir/projects/tir7/user_data/athankar/redteaming/scripts/configs/deploy_game/", config_name="deploy_game.yaml")
def main(cfg: DictConfig):
    # Set up environment
    env = os.environ.copy()
    env["HF_HOME"] = cfg.env.hf_home
    logdir = HydraConfig.get().run.dir
    logger.info("Log dir: {}".format(logdir))
    env["LOGDIR"] = logdir

    processes = []

    # Generate and run controller command
    controller_cmd = base_controller_shell.format(
        controller_port=cfg.ports.controller,
        logdir=logdir
    )
    processes.append(run_background(cfg, controller_cmd, env))

    # Generate and run model worker commands
    for model in cfg.models:
        cmd = base_modelworker_shell.format(
            gpu_id=model.gpu_id,
            model_path=model.path,
            model_name=model.name,
            controller_port=cfg.ports.controller,
            worker_port=model.worker_port,
            device=model.device,
            logdir=logdir,
            out_fname=model.name.replace("/", "_")
        )
        processes.append(run_background(cfg, cmd, env))

    # Generate and run OpenAI API server command
    openai_cmd = base_openai_shell.format(
        controller_port=cfg.ports.controller,
        openai_port=cfg.ports.openai,
        logdir=logdir
    )
    processes.append(run_background(cfg, openai_cmd, env))

    # Sleep for 1 minute to allow services to start up
    time.sleep(60)

    # Check if models are loaded
    if not check_models_loaded(cfg):
        logger.error("Failed to load all models. Exiting.")
        for process in processes:
            process.terminate()
        return

    logger.info(f"All services are running. Logs are being written to: {logdir}")
    logger.info("You can now run your game script separately.")
    logger.info("Remember to terminate these processes when you're done.")

    # # Keep the script running and periodically check if processes are still alive
    # while all(process.poll() is None for process in processes):
    #     time.sleep(60)

    # # If we've exited the loop, at least one process has terminated
    # print("One or more processes have terminated. Shutting down all processes.")
    # for process in processes:
    #     if process.poll() is None:
    #         process.terminate()

    while(1):
        time.sleep(60)

if __name__ == "__main__":
    main()