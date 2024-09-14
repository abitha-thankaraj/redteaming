import glob
import pandas as pd
from redteam.utils.data_utils import read_json


def get_metrics(fname):
    data = read_json(fname)
    # get rewards
    rewards = []
    for d in data:
        rewards.append(d["judge"]['reward'])
        
    
    
    # COnvert to pandas dataframe
