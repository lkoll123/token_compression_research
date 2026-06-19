"""
FastNystromAttention - A drop-in replacement for PyTorch's MultiheadAttention
that uses the Nyström method to approximate attention with better efficiency.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
import torch_quickfps



def _as_batched_bool_mask(
    mask: Optional[torch.Tensor],
    bsz: int,
    N: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return a [bsz, N] boolean mask on `device`."""
    if mask is None:
        return torch.zeros((bsz, N), dtype=torch.bool, device=device)

    mask = mask.to(device=device)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    if mask.ndim == 1:
        if mask.numel() != N:
            raise ValueError(f"{name} must have shape [N] or [B,N]; got {tuple(mask.shape)} vs N={N}")
        mask = mask.unsqueeze(0).expand(bsz, N)
    elif mask.ndim == 2:
        if mask.shape != (bsz, N):
            raise ValueError(f"{name} must have shape [B,N]; got {tuple(mask.shape)} vs {(bsz, N)}")
    else:
        raise ValueError(f"{name} must have shape [N] or [B,N]; got {tuple(mask.shape)}")

    return mask


@torch.no_grad()
def sample_landmarks(
    x: torch.Tensor,
    num_landmarks: int,
    sample_method: str = "fps",          # "fps" or "random"
    guarantee_mask: Optional[torch.Tensor] = None,
    exclude_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample landmark indices from tokens.

    Args:
        x: [B,N,D] or [B,H,N,D]
        num_landmarks: number of indices to return (per batch)
        sample_method: "fps" or "random"
        guarantee_mask: [N] or [B,N] bool, must be included (may be > num_landmarks; will be truncated)
        exclude_mask: [N] or [B,N] bool, must not be included

    Returns:
        idx: [B, num_landmarks] long
    """
    if num_landmarks < 0:
        raise ValueError(f"num_landmarks must be >= 0, got {num_landmarks}")
    bsz = x.shape[0]
    N = x.shape[-2]
    device = x.device

    if num_landmarks == 0:
        return torch.empty((bsz, 0), dtype=torch.long, device=device)

    # Masks -> [B,N] bool
    guarantee_mask = _as_batched_bool_mask(guarantee_mask, bsz, N, device, "guarantee_mask")
    exclude_mask = _as_batched_bool_mask(exclude_mask, bsz, N, device, "exclude_mask")

    # Guarantee wins over exclude (robust to bad inputs)
    exclude_mask = exclude_mask & ~guarantee_mask

    # Points for sampling
    if x.ndim == 4:
        # [B,H,N,D] -> [B,N,H*D] as a view when possible
        points = x.transpose(1, 2).reshape(bsz, N, -1)
    elif x.ndim == 3:
        points = x
    else:
        raise ValueError(f"x must be [B,N,D] or [B,H,N,D], got {tuple(x.shape)}")
    points = points.contiguous()

    restricted_mask = (~guarantee_mask) & (~exclude_mask)
    guarantee_count = guarantee_mask.sum(dim=1)   # [B]
    restricted_count = restricted_mask.sum(dim=1) # [B]

    # How many we *want* to sample from restricted points
    need = (num_landmarks - guarantee_count).clamp(min=0)
    to_sample = torch.minimum(need, restricted_count)  # [B]
    Kmax = int(to_sample.max().item())

    # Batched FPS results (only used for fps mode)
    fps_idx_all: Optional[torch.Tensor] = None
    if sample_method == "fps" and Kmax > 0:
        fps_idx_all = torch_quickfps.sample(
            points,
            Kmax,
            mask=restricted_mask,
            h=8,
            low_d=8,
            return_points=False,
        )
    elif sample_method not in ("fps", "random"):
        raise ValueError(f"Unknown sample_method={sample_method!r}")

    # Assemble final indices directly (no giant scores matrix)
    out = torch.empty((bsz, num_landmarks), dtype=torch.long, device=device)

    for b in range(bsz):
        # 1) guarantee indices (deterministic, sorted by index)
        g = torch.nonzero(guarantee_mask[b], as_tuple=False).squeeze(1)
        if g.numel() > num_landmarks:
            # If user over-guarantees, truncate deterministically
            out[b] = g[:num_landmarks]
            continue

        rem = num_landmarks - int(g.numel())
        picked = [g]

        # 2) fill from restricted set
        if rem > 0:
            if sample_method == "fps" and Kmax > 0 and int(to_sample[b].item()) > 0:
                s = fps_idx_all[b, : int(to_sample[b].item())]
                # defensive filtering (should already be valid):
                s = s[restricted_mask[b, s]]
                if g.numel() > 0:
                    sel = torch.zeros((N,), dtype=torch.bool, device=device)
                    sel[g] = True
                    s = s[~sel[s]]
                s = s[:rem]
                picked.append(s)
                rem -= int(s.numel())

            elif sample_method == "random":
                ridx = torch.nonzero(restricted_mask[b], as_tuple=False).squeeze(1)
                if ridx.numel() > 0:
                    take = min(rem, int(ridx.numel()))
                    perm = torch.randperm(int(ridx.numel()), device=device)[:take]
                    s = ridx[perm]
                    picked.append(s)
                    rem -= int(s.numel())

        idx = torch.cat(picked, dim=0)

        # 3) pad if we still don't have enough (e.g., too much excluded)
        if idx.numel() < num_landmarks:
            avail = torch.nonzero(~exclude_mask[b], as_tuple=False).squeeze(1)  # includes guarantee+restricted
            if avail.numel() == 0:
                pad_val = torch.zeros((), dtype=torch.long, device=device)
                pad = pad_val.expand(num_landmarks - idx.numel())
            else:
                sel = torch.zeros((N,), dtype=torch.bool, device=device)
                sel[idx] = True
                remaining = avail[~sel[avail]]
                if remaining.numel() == 0:
                    pad_val = idx[-1] if idx.numel() > 0 else avail[0]
                    pad = pad_val.expand(num_landmarks - idx.numel())
                else:
                    need_pad = num_landmarks - int(idx.numel())
                    take = remaining[:need_pad]
                    if take.numel() < need_pad:
                        # extend deterministically by repeating last
                        last = take[-1] if take.numel() > 0 else remaining[-1]
                        extra = last.expand(need_pad - int(take.numel()))
                        take = torch.cat([take, extra], dim=0)
                    pad = take
            idx = torch.cat([idx, pad], dim=0)

        out[b] = idx[:num_landmarks]

    return out


def _invert(A: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert a matrix using Newton-Schulz iteration.
    
    This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, 
    leading to faster convergence.
    
    Args:
        A: A square matrix to invert
        
    Returns:
        The inverted matrix
    """
    # Create identity matrix with same dtype and device as input
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    Z = 1 / torch.max(torch.sum(A, dim=-2, keepdim=True), dim=-1, keepdim=True).values * A.mT
    for _ in range(6):
        AZ = A @ Z
        Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))
    return Z


def fast_nystrom_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sample_indices: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_kv_landmarks: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Functional implementation of Fast Nyström Attention.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        sample_indices: Indices of landmark points for Nyström approximation
        attn_mask: Mask to prevent attention to certain positions
        is_causal: Whether to apply causal mask
        
    Returns:
        - Output tensor
    """
    # For self-attention, key and value are the same as query
    #assert key is query and value is query, "Current implementation only supports self-attention"
    
    # Get device
    device = query.device
    
    # Get dimensions
    bsz = query.shape[0]
    bsz_index = torch.arange(bsz, device=device)[:, None]
    head_dim = query.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale
    
    def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        """Helper function to index tensors for landmark selection."""
        return rearrange(t[bsz_index, :, sample_indices, :], "bsz s h d -> bsz h s d")

    # Extract landmark points
    qp, kp = index(query, sample_indices), index(key, sample_indices)

    # Compute Nyström approximation
    A = torch.softmax(scale * (qp @ kp.mT), dim=-1)        # float: [bsz x h x num_sample x num_sample]
    Bv = F.scaled_dot_product_attention(qp, key, value)    # float: [bsz x h x num_sample x d]
    vp = _invert(A) @ Bv                                   # float: [bsz x h x num_sample x d]
    x = F.scaled_dot_product_attention(query, kp, vp)      # float: [bsz x h x N x d]
    
    # # Compute Nyström approximation components
    # Bk = torch.softmax(query @ (scale * kp.mT), dim=-1)
    # Bv = F.scaled_dot_product_attention(qp, key, value)
    # A = rearrange(Bk[bsz_index, :, sample_indices, :], "bsz s1 h s2 -> bsz h s1 s2")
    
    # # Solve the system using the Nyström method
    # vp = _invert(A) @ Bv
    # x = Bk @ vp
    
    if return_kv_landmarks:
        return x, (kp, vp)
    return x


class FastNystromAttention(nn.MultiheadAttention):
    """
    Fast Nyström Attention as a drop-in replacement for PyTorch's MultiheadAttention.
    
    This implementation approximates the attention matrix using the Nyström method,
    which reduces computational complexity for long sequences from O(N²) to O(N),
    where N is the sequence length.
    
    Args:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): If True, adds bias to the projections. Default: True
        add_bias_kv (bool, optional): If True, adds bias to the key and value projections. Default: False
        add_zero_attn (bool, optional): If True, adds a new batch of zeros to the key and value. Default: False
        kdim (int, optional): Dimension of the key. Default: None (=embed_dim)
        vdim (int, optional): Dimension of the value. Default: None (=embed_dim)
        batch_first (bool, optional): If True, input and output tensors are provided as (batch, seq, feature). Default: False
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, 
            add_zero_attn, kdim, vdim, batch_first
        )

    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        sample_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Fast Nyström Attention.
        
        Args:
            query: Query tensor
            key: Key tensor (typically identical to query for self-attention)
            value: Value tensor (typically identical to query for self-attention)
            key_padding_mask: Mask for keys per batch to indicate padding
            need_weights: Whether to return attention weights
            attn_mask: Mask to prevent attention to certain positions
            average_attn_weights: Whether to average attention weights over heads
            is_causal: Whether to apply causal mask
            sample_indices: Indices of landmark points for Nyström approximation
            
        Returns:
            - Output tensor
            - Attention weights if need_weights is True, otherwise None
        """
        # Default to nn.MultiheadAttention if sample_indices is not provided
        if sample_indices is None:
            return super().forward(
                query, key, value, key_padding_mask, need_weights, attn_mask,
                average_attn_weights, is_causal
            )

        # Reshape to [B, N, D] if batch_first=False
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        # Project query, key, value
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        query, key, value = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)
        
        x = fast_nystrom_attention(
            query,
            key,
            value,
            sample_indices,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )
        x = rearrange(x, "b h n d -> b n (h d)")    

        # Project output
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)

        # Reshape back if batch_first=False
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        return x, None