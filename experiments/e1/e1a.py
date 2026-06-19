import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

import random

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class DatasetLoader(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        pixel_values = self.transform(image.convert("RGB"))
        return pixel_values



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment E1")
    parser.add_argument("--results-path", type=str, default="./outputs/results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--titok-model", type=str, default="l32")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int | None) -> int:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else: 
        seed = random.randint(0, 9999)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed



def forward_pass(titok, pixel_values: torch.Tensor):
    enc = titok.encoder
    B = pixel_values.shape[0]
    G = enc.grid_size ** 2

    x = enc.patch_embed(pixel_values)
    x = x.reshape(B, enc.width, -1).permute(0, 2, 1)

    cls = enc.class_embedding.unsqueeze(0).expand(B, 1, enc.width)
    x = torch.cat([cls, x], dim=1)

    pe = enc.positional_embedding.clone().detach().requires_grad_(True)
    x = x + pe.unsqueeze(0).expand(B, -1, -1).to(x.dtype)

    latents = titok.latent_tokens.unsqueeze(0).expand(B, -1, -1)
    latents = latents + enc.latent_token_positional_embedding.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
    x = torch.cat([x, latents], dim=1)

    x = enc.ln_pre(x)
    x = x.permute(1, 0, 2)
    for block in enc.transformer:
        x = block(x)
    x = x.permute(1, 0, 2)

    latent_feats = x[:, 1 + G:, :]
    latent_feats = enc.ln_post(latent_feats)

    return latent_feats, pe


from models import load_titok, extract_titok_features

DATASET_NAME = "imagenet-1k"

def load_data(dataset_name: str=DATASET_NAME, split: str="validation"):
    return load_dataset(dataset_name, split=split)


def run_e1a(args: argparse.Namespace) -> None:
    # Load model
    titok = load_titok(args.titok_model)
    dataset = load_data(DATASET_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    titok.to(device)

    seed = set_seed(args.seed)


    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(256),
        transforms.ToTensor(), 
    ])

    dataset_wrapped = DatasetLoader(dataset, transform)
    data_loader = DataLoader(dataset_wrapped, batch_size=args.batch_size, shuffle=True)

    batch_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing batches")
    for idx, batch in batch_bar:
        batch = batch.to(device)
        latents, pe = forward_pass(titok, batch)




    print(dataset[0])



if __name__ == "__main__":
    args = parse_args()
    run_e1a(args=args)