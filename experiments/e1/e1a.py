import torch
import sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from models import load_titok, extract_titok_features

DATASET_NAME = "imagenet-1k"

def load_data(dataset_name: str=DATASET_NAME, split: str="validation"):
    return load_dataset(dataset_name, split=split)


def run_e1a(model_key: str, results_path: str="./outputs/results") -> None:
    # Load model
    titok = load_titok(model_key)
    dataset = load_data(DATASET_NAME)

    print(dataset[0])


    # Load dataset (e.g., ImageNet validation set)
    # dataset = ...

    # Extract features and evaluate on downstream tasks
    # for batch in dataset:
    #     pixel_values = batch["pixel_values"]
    #     features = extract_titok_features(titok, pixel_values)
    #     # Evaluate features on downstream tasks (e.g., classification, retrieval)
    #     ...


if __name__ == "__main__":
    run_e1a(model_key="l32")