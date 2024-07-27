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


def get_harmbench_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 400 prompts from the HarmBench dataset.
    Paper: https://arxiv.org/abs/2402.04249
    Data: https://github.com/centerforaisafety/HarmBench

    Example Usage:
        dataset = get_harmbench_dataset(dataset_path)
        for i in range(len(dataset)):
            print(dataset[i]['prompt'])
    """
    return load_dataset(
        "csv", 
        data_files=dataset_path, 
        split="train",
    ).rename_column("Behavior", "prompt")


def get_dataset(
    dataset_name: str,
    dataset_path: str,
) -> ArrowDataset:
    """
    Wrapper function unifying all the individual dataset loading functions
    """

    dataset_loading_functions = {
        "openai": get_openai_redteaming_dataset,
        "harmbench": get_harmbench_dataset,
    }

    dataset_loading_kwargs = {
        "dataset_path": dataset_path,
    }

    if dataset_name not in dataset_loading_functions:
        raise ValueError(f"Give dataset {dataset_name} not supported.")

    return dataset_loading_functions[dataset_name](**dataset_loading_kwargs)
