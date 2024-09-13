import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(PARENT_DIR, "data")
SCRIPTS_DIR = os.path.join(PARENT_DIR, "scripts")
SCRIPTS_CONFIG_DIR = os.path.join(SCRIPTS_DIR, "configs")
EXPERIMENT_SCRIPTS_DIR = os.path.join(SCRIPTS_DIR, "experiments")
EXPERIMENT_LOG_DIR = os.path.join(SCRIPTS_DIR, "logs")
EXPERIMENT_CONFIG_DIR = os.path.join(EXPERIMENT_SCRIPTS_DIR, "configs")
DATAGEN_CONFIG_DIR = os.path.join(SCRIPTS_DIR, "configs")


WANDB_DIR = EXPERIMENT_LOG_DIR
