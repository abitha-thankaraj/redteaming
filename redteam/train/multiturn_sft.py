from redteam.utils.data_utils import read_json

def load_conversations(fname, keys_to_keep):
    data = read_json(fname)
    filtered_data = list(map(lambda d: {k: d[k] for k in keys_to_keep if k in d}, data))
    conversations = list(map(lambda d: d["conversation"], filtered_data))

    # convert conversations to template
    

