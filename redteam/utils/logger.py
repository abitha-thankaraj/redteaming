import wandb
import logging
import sys
import os
import omegaconf
from hydra.core.hydra_config import HydraConfig

class Logger:
    def __init__(self, simple_log = None, log_dir = None) -> None:
        self.simple_log = simple_log
        
        if log_dir is not None:
            self._configure_simple_log(log_dir)

    def _configure_simple_log(self, log_dir):
        self.simple_log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.simple_log.addHandler(file_handler)
        self.simple_log.addHandler(stdout_handler)
    
    def log(self, msg):
        # if self.log_wandb:
        #     if type(msg) is dict:
        #         wandb.log(msg)
            # else:
            #     raise Warning('The logger can currently only log dictionaries to wandb')
        self.simple_log.info(msg)