import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import json

import random

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from models import load_titok


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
    parser.add_argument("--visualize_indices", type=int, nargs="+", default=[0])
    parser.add_argument("--min-index", type=int, default=0)
    parser.add_argument("--max-index", type=int, default=-1)
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


def visualize_gradients(idx: int, latent_feats: torch.Tensor, pe: torch.Tensor, args: argparse.Namespace) -> None:
    # Compute gradients

    output_path = Path(args.results_path) / "e1a"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"gradient_visualization_{idx}.png"

    k = latent_feats.shape[0]
    grid_size = int((pe.shape[0] - 1) ** 0.5)

    sensitivities = torch.zeros(k, grid_size ** 2)

    for i, latent in enumerate(latent_feats):
        if pe.grad is not None:
            pe.grad.zero_()
        latent.pow(2).sum().backward(retain_graph=True)
        pe_grad = pe.grad
        sensitivities[i] = pe_grad[1:].norm(dim=-1)

    sensitivities = sensitivities.reshape(k, grid_size, grid_size).detach().cpu()

    x, y = 0, 0
    match args.titok_model:
        case "l32":
            x, y = 4, 8
        case "b64":
            x, y = 4, 16
        case "s128": 
            x, y = 4, 32
        

    

    fig, axes = plt.subplots(x, y, figsize=(20, 20))

    for i, axis in enumerate(axes.flatten()):
        if i < k:
            axis.imshow(sensitivities[i])
            axis.set_title(f"Latent Token {i}")

        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def compute_entropy(latents_feats: torch.Tensor, pe: torch.Tensor) -> (List[float], List[float]):

    B, K, W = latents_feats.shape
    G = pe.shape[0] - 1
    
    entropies_a = []
    entropies_b = []

    for b in range(B):
        latents = latents_feats[b]
        sensitivities = torch.zeros(K, G)
        for k in range(K):
            if pe.grad is not None:
                pe.grad.zero_()

            latents[k].pow(2).sum().backward(retain_graph=True)
            pe_grad = pe.grad
            sensitivities[k] = pe_grad[1:].norm(dim=-1)

        # Option A: collective entropy (sum across latents first)
        collective = sensitivities.sum(dim=0)                     # (G,)
        probs_a = collective / collective.sum()
        entropy_a = -(probs_a * (probs_a + 1e-12).log()).sum()
        entropies_a.append(entropy_a.item())

        # Option B: per-latent entropy, averaged
        probs_b = sensitivities / sensitivities.sum(dim=1, keepdim=True)  # (K, G)
        entropy_per_latent = -(probs_b * (probs_b + 1e-12).log()).sum(dim=1)  # (K,)
        entropies_b.append(entropy_per_latent.mean().item())

    return entropies_a, entropies_b

        




    


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

    visualize_indices = set(args.visualize_indices)


    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(256),
        transforms.ToTensor(), 
    ])

    min_index = args.min_index if 0 <= args.min_index < len(dataset) else 0
    max_index = args.max_index if args.min_index < args.max_index <= len(dataset) else len(dataset)

    dataset = dataset.select(range(min_index, max_index))

    dataset_wrapped = DatasetLoader(dataset, transform)
    data_loader = DataLoader(dataset_wrapped, batch_size=args.batch_size, shuffle=True)

    for idx in visualize_indices:
        batch = dataset_wrapped[idx].to(device).unsqueeze(0)
        latent_feats, pe = forward_pass(titok, batch)
        visualize_gradients(idx, latent_feats[0], pe, args)

        original_img = dataset_wrapped.dataset[idx]["image"]
        output_dir = Path(args.results_path) / "e1a"
        output_dir.mkdir(parents=True, exist_ok=True)
        original_img.save(output_dir / f"original_{idx}.png")

    collective_entropy: list[float] = []
    avg_entropy_across_latents: list[float] = []

    batch_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing batches")
    for idx, batch in batch_bar:
        batch = batch.to(device)
        latents_batch, pe = forward_pass(titok, batch)
        entropies_a, entropies_b = compute_entropy(latents_batch, pe)
        collective_entropy += entropies_a
        avg_entropy_across_latents += entropies_b

    collective_entropy_mean = sum(collective_entropy) / len(collective_entropy) if collective_entropy else 0.0
    avg_entropy_across_latents_mean = sum(avg_entropy_across_latents) / len(avg_entropy_across_latents) if avg_entropy_across_latents else 0.0

    output_dir = Path(args.results_path) / "e1a"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"entropy_results.json"

    results = {
        "seed": seed,
        "titok_model": args.titok_model,
        "collective_entropy_mean": collective_entropy_mean,
        "avg_entropy_across_latents_mean": avg_entropy_across_latents_mean,
        "collective_entropy": collective_entropy,
        "avg_entropy_across_latents": avg_entropy_across_latents
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)




if __name__ == "__main__":
    args = parse_args()
    run_e1a(args=args)