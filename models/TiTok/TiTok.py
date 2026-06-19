from __future__ import annotations
import sys
from pathlib import Path
import torch

# add 1d-tokenizer to path so modeling.titok is importable
TITOK_REPO = Path(__file__).resolve().parent / "1d-tokenizer"
if str(TITOK_REPO) not in sys.path:
    sys.path.insert(0, str(TITOK_REPO))

from modeling.titok import TiTok

MODELS = {
    "l32": "yucornetto/tokenizer_titok_l32_imagenet",
    "b64": "yucornetto/tokenizer_titok_b64_imagenet",
    "s128": "yucornetto/tokenizer_titok_s128_imagenet",
}

def load_titok(model_key: str = "l32") -> TiTok:
    return TiTok.from_pretrained(MODELS[model_key]).eval()


@torch.no_grad()
def extract_titok_features(titok: TiTok, pixel_values: torch.Tensor) -> torch.Tensor:
    enc = titok.encoder
    B = pixel_values.shape[0]
    G = enc.grid_size ** 2
    x = enc.patch_embed(pixel_values)
    x = x.reshape(B, enc.width, -1).permute(0, 2, 1)
    cls = enc.class_embedding.unsqueeze(0).expand(B, 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + enc.positional_embedding.to(x.dtype)
    latents = titok.latent_tokens.unsqueeze(0).expand(B, -1, -1)
    latents = latents + enc.latent_token_positional_embedding.to(x.dtype)
    x = torch.cat([x, latents], dim=1)
    x = enc.ln_pre(x)
    x = x.permute(1, 0, 2)
    for block in enc.transformer:
        x = block(x)
    x = x.permute(1, 0, 2)
    latent_feats = x[:, 1 + G:, :]
    latent_feats = enc.ln_post(latent_feats)
    return latent_feats  # (B, K, width)