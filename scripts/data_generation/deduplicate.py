import os
import json
import hashlib
from redteam.utils.data_utils import read_json, write_json

# def hash_json(json_obj):
#     # Convert JSON object to a sorted, canonical string representation
#     canonical_str = json.dumps(json_obj, sort_keys=True)
#     # Create a hash of the string
#     return hashlib.md5(canonical_str.encode('utf-8')).hexdigest()
def hash_json(json_obj):
    return hash(str(json_obj))
    
def deduplicate_json_list(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary to store unique items
    unique_items = {}

    # Iterate through the list and add unique items to the dictionary
    for item in data:
        item_hash = hash_json(item)
        if item_hash not in unique_items:
            unique_items[item_hash] = item

    # Convert the dictionary values back to a list
    deduplicated_data = list(unique_items.values())

    # Write the deduplicated data back to a new JSON file
    output_file_path = file_path.replace("combined/", "deduplicated/").rsplit('.', 1)[0] + '_deduplicated.json'
    
    # with open(output_file_path, 'w') as file:
    #     json.dump(deduplicated_data, file, indent=2)
    write_json(deduplicated_data, output_file_path)
    
    print(len(data), len(deduplicated_data))
    print(f"Deduplicated data written to {output_file_path}")
    print(f"Original count: {len(data)}, Deduplicated count: {len(deduplicated_data)}")

# Usage
if __name__ == "__main__":
    # FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/mistral/combined"
    FOLDER = "/data/group_data/rl/datasets/redteaming/best_of_n/llama/combined"

    for fname in os.listdir(FOLDER):
        if fname.endswith('.json'):
            print(fname)
            file_path = os.path.join(FOLDER, fname)
            deduplicate_json_list(file_path)
