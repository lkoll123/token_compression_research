"""Token Compression for Llava-Next - Pytorch encoder module for compressing image tokens into a 1D latent space"""
from .pruning import LlavaNextCompressor

__all__ = ["LlavaNextCompressor"]