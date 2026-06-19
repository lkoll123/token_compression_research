from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import warnings

import torch
import torch.nn as nn

from .attention import sample_landmarks, fast_nystrom_attention


def copy_non_module_attributes(source: nn.Module, destination: nn.Module):
    """
    Copies attributes from a source nn.Module to a destination nn.Module,
    skipping any attributes that are themselves nn.Module.
    """
    for attr_name, attr_value in source.__dict__.items():
        if attr_name != '_modules':
            setattr(destination, attr_name, attr_value)


class FNACacheMixin:
    def load_cache(self, d: Dict[str, Any]) -> None:
        self.fna_cache.clear()
        self.fna_cache.update(d)

    def update_cache(self, d: Dict[str, Any]) -> None:
        self.fna_cache.update(d)


VALID_SAMPLING_STRATEGIES = {"fps", "random"}
VALID_SAMPLING_FEATURES = {"input", "q", "k", "v"}


def normalize_fna_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config) if config is not None else {}

    if "resample_every_layer" not in cfg and "resample_fps" in cfg:
        warnings.warn(
            "fna_config['resample_fps'] is deprecated; use 'resample_every_layer' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg["resample_every_layer"] = bool(cfg["resample_fps"])
        del cfg["resample_fps"]

    cfg.setdefault("resample_every_layer", False)

    sampling_strategy = str(cfg.get("sampling_strategy", "fps")).lower()
    if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unsupported sampling_strategy '{sampling_strategy}'. Choose from {sorted(VALID_SAMPLING_STRATEGIES)}"
        )
    cfg["sampling_strategy"] = sampling_strategy

    sampling_features = str(cfg.get("sampling_features", "input")).lower()
    if sampling_features not in VALID_SAMPLING_FEATURES:
        raise ValueError(
            f"Unsupported sampling_features '{sampling_features}'. Choose from {sorted(VALID_SAMPLING_FEATURES)}"
        )
    cfg["sampling_features"] = sampling_features

    cfg.setdefault("fna_layers", [])
    cfg.setdefault("num_sample", 0)

    return cfg


def _resolve_qkv_sample_indices(
    sample_indices: Optional[torch.Tensor],
    sampling_features: Optional[str],
    num_sample: Optional[int],
    sampling_strategy: Optional[str],
    guarantee_mask: Optional[torch.Tensor],
    exclude_mask: Optional[torch.Tensor],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    if sample_indices is not None:
        return sample_indices
    if sampling_features not in {"q", "k", "v"}:
        return None
    if num_sample is None:
        raise ValueError("num_sample must be provided when sampling_features is q/k/v.")

    source_map = {
        "q": query_states,
        "k": key_states,
        "v": value_states,
    }
    sampling_source = source_map[sampling_features].mean(dim=1)
    #qkv_mean = torch.stack((query_states, key_states, value_states), dim=0).mean(dim=0)
    #sampling_source = qkv_mean.mean(dim=1)
    return sample_landmarks(
        sampling_source,
        num_sample,
        sample_method=sampling_strategy or "fps",
        guarantee_mask=guarantee_mask,
        exclude_mask=exclude_mask,
    )


def _resolve_layer_sample_indices(
    *,
    sampling_features: str,
    resample_every_layer: bool,
    sample_indices: Optional[torch.Tensor],
    fna_cache: Dict[str, Any],
    hidden_states: torch.Tensor,
    num_sample: int,
    sampling_strategy: str,
    mask_dict: Dict[str, Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    if sampling_features in {"q", "k", "v"} and not resample_every_layer:
        cached = fna_cache.get("sample_indices")
        if cached is not None:
            sample_indices = cached

    if sampling_features == "input" and (resample_every_layer or sample_indices is None):
        sample_indices = sample_landmarks(
            hidden_states,
            num_sample,
            sample_method=sampling_strategy,
            guarantee_mask=mask_dict.get("guarantee"),
            exclude_mask=mask_dict.get("exclude"),
        )

    return sample_indices


def _update_cached_sample_indices(
    *,
    sampling_features: str,
    resample_every_layer: bool,
    fna_cache: Dict[str, Any],
    attention_module: nn.Module,
    sample_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if resample_every_layer or sampling_features not in {"q", "k", "v"}:
        return sample_indices

    last = getattr(attention_module, "last_sample_indices", sample_indices)
    if last is not None:
        fna_cache["sample_indices"] = last
    return last


# --------------------
# CLIP
# --------------------
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPModel,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_2


class CLIPFastNystromAttention(CLIPAttention):
    @ classmethod
    def from_clip_attention(
        cls, 
        attention: CLIPAttention, 
        config: CLIPConfig,
    ) -> "CLIPFastNystromAttention":
        fna_attention = cls(config)
        copy_non_module_attributes(attention, fna_attention)
        fna_attention.load_state_dict(attention.state_dict())
        return fna_attention
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_fna: Optional[bool] = False,
        sample_indices: Optional[torch.Tensor] = None,
        sampling_features: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
        num_sample: Optional[int] = None,
        guarantee_mask: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:    
        if not use_fna:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
        if sample_indices is None and sampling_features not in {"q", "k", "v"}:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )

        if output_attentions:
            raise NotImplementedError("Output attentions are not implemented for FastNystromAttention.")

        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if not is_torch_greater_or_equal_than_2_2 and query_states.device.type == "cuda" and attn_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        sample_indices = _resolve_qkv_sample_indices(
            sample_indices=sample_indices,
            sampling_features=sampling_features,
            num_sample=num_sample,
            sampling_strategy=sampling_strategy,
            guarantee_mask=guarantee_mask,
            exclude_mask=exclude_mask,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )

        if sample_indices is None:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
        self.last_sample_indices = sample_indices

        # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
        attn_output = fast_nystrom_attention(
            query_states,
            key_states,
            value_states,
            sample_indices=sample_indices,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class CLIPEncoderLayerFNA(CLIPEncoderLayer):
    @ classmethod
    def from_clip_encoder_layer(
        cls,
        layer: CLIPEncoderLayer,
        config: CLIPConfig,
    ) -> "CLIPEncoderLayerFNA":
        fna_layer = cls(config)
        copy_non_module_attributes(layer, fna_layer)
        fna_layer.load_state_dict(layer.state_dict())
        return fna_layer

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        # Replace CLIPAttention with CLIPFastNystromAttention
        self.self_attn = CLIPFastNystromAttention.from_clip_attention(self.self_attn, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        use_fna: Optional[bool] = False,
        sample_indices: Optional[torch.Tensor] = None,
        sampling_features: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
        num_sample: Optional[int] = None,
        guarantee_mask: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            use_fna=use_fna,
            sample_indices=sample_indices,
            sampling_features=sampling_features,
            sampling_strategy=sampling_strategy,
            num_sample=num_sample,
            guarantee_mask=guarantee_mask,
            exclude_mask=exclude_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CLIPEncoderFNA(CLIPEncoder):
    @ classmethod
    def from_clip_encoder(
        cls,
        encoder: CLIPEncoder,
        config: CLIPConfig,
        fna_config: Dict[str, Any] = {},
        fna_cache: Dict[str, Any] = {}
    ) -> "CLIPEncoderFNA":
        fna_encoder = cls(config, fna_config, fna_cache)
        copy_non_module_attributes(encoder, fna_encoder)
        fna_encoder.load_state_dict(encoder.state_dict())
        return fna_encoder

    def __init__(
        self, 
        config: CLIPConfig, 
        fna_config: Optional[Dict[str, Any]] = {}, 
        fna_cache: Optional[Dict[str, Any]] = {}
    ):
        super().__init__(config)
        self.fna_config = normalize_fna_config(fna_config)
        self.fna_cache = fna_cache

        # Replace CLIPEncoderLayer with CLIPEncoderLayerFNA
        for idx, encoder_layer in enumerate(self.layers):
            self.layers[idx] = CLIPEncoderLayerFNA.from_clip_encoder_layer(encoder_layer, config)

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutput:
        # Always resample per batch by clearing cached indices
        self.fna_cache.pop("sample_indices", None)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        sample_indices = None
        mask_dict = self.fna_cache.get("mask_dict", {})
        fna_layers = self.fna_config["fna_layers"]
        sampling_features = self.fna_config["sampling_features"]
        sampling_strategy = self.fna_config["sampling_strategy"]
        num_sample = self.fna_config["num_sample"]
        resample_every_layer = self.fna_config["resample_every_layer"]
        for idx, encoder_layer in enumerate(self.layers):
            layer_args = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "causal_attention_mask": causal_attention_mask,
                "output_attentions": output_attentions
            }
            if idx in fna_layers:
                sample_indices = _resolve_layer_sample_indices(
                    sampling_features=sampling_features,
                    resample_every_layer=resample_every_layer,
                    sample_indices=sample_indices,
                    fna_cache=self.fna_cache,
                    hidden_states=hidden_states,
                    num_sample=num_sample,
                    sampling_strategy=sampling_strategy,
                    mask_dict=mask_dict,
                )
                layer_args["use_fna"] = True
                layer_args["sample_indices"] = sample_indices
                layer_args["sampling_features"] = sampling_features
                layer_args["sampling_strategy"] = sampling_strategy
                layer_args["num_sample"] = num_sample
                layer_args["guarantee_mask"] = mask_dict.get("guarantee")
                layer_args["exclude_mask"] = mask_dict.get("exclude")

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__, 
                    **layer_args,
                )
            else:
                layer_outputs = encoder_layer(**layer_args)

            hidden_states = layer_outputs[0]

            if idx in fna_layers:
                sample_indices = _update_cached_sample_indices(
                    sampling_features=sampling_features,
                    resample_every_layer=resample_every_layer,
                    fna_cache=self.fna_cache,
                    attention_module=encoder_layer.self_attn,
                    sample_indices=sample_indices,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPModelFNA(CLIPModel, FNACacheMixin):
    def __init__(
        self, 
        config: CLIPConfig, 
        fna_config: Optional[Dict[str, Any]] = {}, 
        fna_cache: Optional[Dict[str, Any]] = {}
    ):
        super().__init__(config)
        self.fna_config = normalize_fna_config(fna_config)
        self.fna_cache = fna_cache

        # Replace CLIPEncoder with CLIPEncoderFNA
        self.vision_model.encoder = CLIPEncoderFNA.from_clip_encoder(
            self.vision_model.encoder, 
            config.vision_config, 
            self.fna_config, 
            fna_cache
        )


# --------------------
# DINOv2
# --------------------



# --------------------
# LLaVA-NeXT
# --------------------
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
)

from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextCausalLMOutputWithPast,
    LlavaNextConfig,
)


class LlamaFastNystromAttention(LlamaAttention):
    @ classmethod
    def from_llama_attention(
        cls, 
        attention: LlamaAttention, 
        config: LlamaConfig,
        layer_idx: int
    ) -> "LlamaFastNystromAttention":
        fna_attention = cls(config, layer_idx)
        copy_non_module_attributes(attention, fna_attention)
        fna_attention.load_state_dict(attention.state_dict())
        return fna_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_fna: Optional[bool] = False,
        sample_indices: Optional[torch.Tensor] = None,
        sampling_features: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
        num_sample: Optional[int] = None,
        guarantee_mask: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not use_fna:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        if sample_indices is None and sampling_features not in {"q", "k", "v"}:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kwargs.get("output_attentions", False):
            raise NotImplementedError("Output attentions are not implemented for FastNystromAttention.")

        sample_indices = _resolve_qkv_sample_indices(
            sample_indices=sample_indices,
            sampling_features=sampling_features,
            num_sample=num_sample,
            sampling_strategy=sampling_strategy,
            guarantee_mask=guarantee_mask,
            exclude_mask=exclude_mask,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )

        if sample_indices is None:
            self.last_sample_indices = None
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        self.last_sample_indices = sample_indices
        
        attn_output, (key_landmarks, value_landmarks) = fast_nystrom_attention(
            query_states,
            key_states,
            value_states,
            sample_indices=sample_indices,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
            return_kv_landmarks=True,
        )
        
        # Cache the landmarks for future text generation
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_value.update(key_landmarks, value_landmarks, self.layer_idx, cache_kwargs)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    

class LlamaDecoderLayerFNA(LlamaDecoderLayer):
    @ classmethod
    def from_llama_decoder_layer(
        cls, 
        layer: LlamaDecoderLayer, 
        config: LlamaConfig,
        layer_idx: int
    ) -> "LlamaDecoderLayerFNA":
        fna_layer = cls(config, layer_idx)
        copy_non_module_attributes(layer, fna_layer)
        fna_layer.load_state_dict(layer.state_dict())
        return fna_layer
    
    def __init__(
        self,
        config: LlamaConfig, 
        layer_idx: int
    ):
        super().__init__(config, layer_idx)

        # Replace LlamaAttention with LlamaFastNystromAttention
        self.self_attn = LlamaFastNystromAttention.from_llama_attention(self.self_attn, config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        use_fna: Optional[bool] = False,
        sample_indices: Optional[torch.Tensor] = None,
        sampling_features: Optional[str] = None,
        sampling_strategy: Optional[str] = None,
        num_sample: Optional[int] = None,
        guarantee_mask: Optional[torch.Tensor] = None,
        exclude_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            use_fna=use_fna,
            sample_indices=sample_indices,
            sampling_features=sampling_features,
            sampling_strategy=sampling_strategy,
            num_sample=num_sample,
            guarantee_mask=guarantee_mask,
            exclude_mask=exclude_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    

class LlamaModelFNA(LlamaModel):
    @ classmethod
    def from_llama_model(
        cls,
        model: LlamaModel, 
        config: LlamaConfig,
        fna_config: Dict[str, Any] = {}, 
        fna_cache: Dict[str, Any] = {}
    ) -> "LlamaModelFNA":
        normalized_config = normalize_fna_config(fna_config)
        fna_model = cls(config, normalized_config, fna_cache)
        copy_non_module_attributes(model, fna_model)
        fna_model.load_state_dict(model.state_dict())
        return fna_model

    def __init__(
        self, 
        config: LlamaConfig, 
        fna_config: Optional[Dict[str, Any]] = {}, 
        fna_cache: Optional[Dict[str, Any]] = {}
    ):
        super().__init__(config)
        self.fna_config = normalize_fna_config(fna_config)
        self.fna_cache = fna_cache

        # Replace LlamaDecoderLayer with LlamaDecoderLayerFNA
        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            self.layers[idx] = LlamaDecoderLayerFNA.from_llama_decoder_layer(decoder_layer, config, idx)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        has_image_tokens: Optional[bool] = False,
        n_image_tokens: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        # Always resample per batch by clearing cached indices
        self.fna_cache.pop("sample_indices", None)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        sample_indices = None
        mask_dict = self.fna_cache.get("mask_dict", {})
        fna_layers = self.fna_config["fna_layers"]
        sampling_features = self.fna_config["sampling_features"]
        sampling_strategy = self.fna_config["sampling_strategy"]
        num_sample = self.fna_config["num_sample"]
        resample_every_layer = self.fna_config["resample_every_layer"]
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_args = {
                "hidden_states": hidden_states,
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_values,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
            }
            
            if has_image_tokens and layer_idx in fna_layers:
                sample_indices = _resolve_layer_sample_indices(
                    sampling_features=sampling_features,
                    resample_every_layer=resample_every_layer,
                    sample_indices=sample_indices,
                    fna_cache=self.fna_cache,
                    hidden_states=hidden_states,
                    num_sample=num_sample,
                    sampling_strategy=sampling_strategy,
                    mask_dict=mask_dict,
                )
                layer_args["use_fna"] = True
                layer_args["sample_indices"] = sample_indices
                layer_args["sampling_features"] = sampling_features
                layer_args["sampling_strategy"] = sampling_strategy
                layer_args["num_sample"] = num_sample
                layer_args["guarantee_mask"] = mask_dict.get("guarantee")
                layer_args["exclude_mask"] = mask_dict.get("exclude")

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    **layer_args,
                )
            else:
                layer_outputs = decoder_layer(**layer_args, **flash_attn_kwargs)

            hidden_states = layer_outputs[0]

            if has_image_tokens and layer_idx in fna_layers:
                sample_indices = _update_cached_sample_indices(
                    sampling_features=sampling_features,
                    resample_every_layer=resample_every_layer,
                    fna_cache=self.fna_cache,
                    attention_module=decoder_layer.self_attn,
                    sample_indices=sample_indices,
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        

class LlavaNextForConditionalGenerationFNA(LlavaNextForConditionalGeneration, FNACacheMixin):
    def __init__(
        self, 
        config: LlavaNextConfig, 
        fna_config: Optional[Dict[str, Any]] = {}, 
        fna_cache: Optional[Dict[str, Any]] = {}
    ):
        super().__init__(config)
        normalized_config = normalize_fna_config(fna_config)
        self.fna_config = normalized_config
        self.fna_cache = fna_cache

        # Replace LlamaModel with LlamaModelFNA
        self.language_model.model = LlamaModelFNA.from_llama_model(
            self.language_model.model, 
            config.text_config, 
            normalized_config, 
            fna_cache
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        if inputs_embeds is not None:
            raise NotImplementedError('inputs_embeds is not supported in LlavaNextFNA. Please use input_ids instead.')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = int((input_ids == self.config.image_token_index).sum().item())
                n_image_features = int(image_features.shape[0])
                if getattr(self, "_visionzip_truncate_features", False) and n_image_features > n_image_tokens:
                    logging.warning(
                        "VisionZip: truncating image features from %d to %d to match image tokens.",
                        n_image_features,
                        n_image_tokens,
                    )
                    image_features = image_features[:n_image_tokens]
                else:
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens: {n_image_tokens}, features {n_image_features}"
                    )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            has_image_tokens = True
            n_image_tokens = special_image_mask[:, :, 0].sum(dim=1)
            # TODO: This seems to hurt performance, why?
            self.fna_cache["mask_dict"] = {
                "guarantee": ~special_image_mask[:, :, 0].bool(), # Guarantee mask for non-image tokens
                "exclude": None,
            }
        else:
            has_image_tokens = False
            n_image_tokens = None

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            has_image_tokens=has_image_tokens,
            n_image_tokens=n_image_tokens,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


# --------------------
# StableDiffusion3
# --------------------