from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset


def get_openai_redteaming_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 15 (the actual dataset contains 16 examples, we omit one) prompts
    from GPT-4 Technical Report, used to redteam the original GPT-4 model.

    Example Usage:
        dataset = get_openai_redteaming_dataset(dataset_path)
        for i in range(len(dataset)):
            print(dataset[i]['prompt'])
    """
    return load_dataset(
        "json", 
        data_files=dataset_path, 
        field="Questions", 
        split='train',
    )
