from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset


def get_dataset(
    dataset_name: str,
    dataset_path: str = None,
) -> ArrowDataset:
    """
    Wrapper function unifying all the individual dataset loading functions

    Example Usage:
        dataset = get_dataset(
            dataset_name="jailbreakbench",
            dataset_path="",
        )
        for i in range(len(dataset)):
            print(dataset[i]['prompt'])
    """
    dataset_loading_functions = {
        "openai": get_openai_redteaming_dataset,
        "harmbench": get_harmbench_dataset,
        "advbench": get_advbench_dataset,
        "jailbreakbench": get_jailbreakbench_dataset,
        "ultrasafety": get_ultrasafety_dataset,
        "simplesafetytests": get_simplesafetytests_dataset,
    }

    dataset_loading_kwargs = {
        "dataset_path": dataset_path,
    }

    if dataset_name not in dataset_loading_functions:
        raise ValueError(f"Give dataset {dataset_name} not supported.")

    return dataset_loading_functions[dataset_name](**dataset_loading_kwargs)


def get_harmbench_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 400 prompts from the HarmBench dataset.
    Paper: https://arxiv.org/abs/2402.04249
    Data: https://github.com/centerforaisafety/HarmBench
    """
    return load_dataset(
        "csv",
        data_files=dataset_path,
        split="train",
    ).rename_column("Behavior", "prompt")


def get_advbench_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 500 harmful prompts from the AdvBench dataset.
    Paper: https://arxiv.org/abs/2307.15043
    Data: https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench
    """
    return load_dataset("csv", data_files=dataset_path, split="train").rename_column(
        "goal", "prompt"
    )


def get_ultrasafety_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 3k harmful prompts from the UltraSafety dataset.
    Paper: https://arxiv.org/abs/2402.19085
    Data: https://huggingface.co/datasets/openbmb/UltraSafety
    """
    return (
        load_dataset("openbmb/UltraSafety", split="train")
        .rename_column("prompt", "old_prompt")
        .rename_column("instruction", "prompt")
    )


# Eval datasets

def get_openai_redteaming_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 15 (the actual dataset contains 16 examples, we omit one) prompts
    from GPT-4 Technical Report, used to redteam the original GPT-4 model.
    """
    return load_dataset(
        "json",
        data_files=dataset_path,
        field="Questions",
        split="train",
    )


def get_jailbreakbench_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 100 harmful prompts from the JailBreakBench dataset.
    Paper: https://arxiv.org/abs/2404.01318
    Data: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/viewer/judge_comparison
    """
    return load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
    ).rename_column("Goal", "prompt")


def get_simplesafetytests_dataset(dataset_path: str) -> ArrowDataset:
    """
    Loads the 100 harmful prompts from the SimpleSafetyTests dataset.
    Paper: https://arxiv.org/abs/2311.08370
    Data: https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests
    """
    return load_dataset("Bertievidgen/SimpleSafetyTests", split="test")
