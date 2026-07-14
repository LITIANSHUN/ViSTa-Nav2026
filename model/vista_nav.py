from __future__ import annotations

"""ViSTa-Nav model components.

This module provides a reviewer-facing implementation skeleton for the model
presented in the manuscript.  It intentionally separates the three frozen
visual foundation models used by ViSTa-Nav:

1. CLIP ViT-B/32: global image-text semantic alignment;
2. DINOv2: patch-level historical visual memory;
3. Stable-Diffusion VAE: compact latent visual foresight targets.

The adapters use lazy optional imports so that the core trajectory model can
still be inspected, unit-tested, and executed in an offline review environment
without downloading large pretrained weights.  When the required packages and
checkpoints are available, setting ``allow_fallback=False`` enforces the exact
pretrained path and raises a clear error if a dependency is missing.
"""

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Reproducibility-friendly fallback encoders
# -----------------------------------------------------------------------------


class HashTextEncoder(nn.Module):
    """Deterministic offline text encoder used only as a dependency fallback.

    The paper uses a frozen CLIP ViT-B/32 text encoder.  This class preserves
    the same batched interface when OpenCLIP/CLIP weights are unavailable, so
    reviewers can still verify tensor shapes and the complete training graph.
    It is *not* intended to reproduce the reported semantic performance.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, texts: Sequence[str], device: torch.device) -> Tensor:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(list(digest), dtype=torch.float32)
            values = values.repeat((self.dim + len(values) - 1) // len(values))[: self.dim]
            rows.append((values - 127.5) / 127.5)
        return torch.stack(rows).to(device)


class ConvImageFallback(nn.Module):
    """Small deterministic image encoder used when pretrained models are absent."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


# -----------------------------------------------------------------------------
# Foundation-model adapters
# -----------------------------------------------------------------------------


@dataclass
class VisualAdapterConfig:
    """Configuration for frozen visual foundation-model adapters.

    The defaults follow the manuscript: CLIP ViT-B/32 for global image-text
    alignment, DINOv2 for patch-level temporal memory, and the Stable Diffusion
    VAE encoder for visual foresight supervision.
    """

    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    dino_model_name: str = "dinov2_vitb14"
    sd_vae_model_name: str = "stabilityai/sd-vae-ft-mse"
    freeze_backbones: bool = True
    allow_fallback: bool = True
    clip_input_size: int = 224
    dino_input_size: int = 224
    vae_scaling_factor: float = 0.18215


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class CLIPViTB32Adapter(nn.Module):
    """Adapter for frozen CLIP ViT-B/32 image and text encoders.

    Role in ViSTa-Nav:
        CLIP supplies global semantic descriptors for the current RGB view and
        the natural-language navigation instruction.  These descriptors are
        projected to the shared model dimension and fused by cross-attention,
        corresponding to the semantic branch in Eqs. (9)--(11).

    Expected image input:
        Tensor of shape ``[B, 3, H, W]`` in ``[0, 1]``.  The adapter performs
        CLIP normalization internally.  Text input is a sequence of strings.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, output_dim: int, cfg: VisualAdapterConfig):
        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.backend = "fallback"
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.image_fallback = ConvImageFallback(512)
        self.text_fallback = HashTextEncoder(512)
        native_dim = 512  # CLIP ViT-B/32 embedding dimension

        try:
            import open_clip  # type: ignore

            model, _, _ = open_clip.create_model_and_transforms(
                cfg.clip_model_name,
                pretrained=cfg.clip_pretrained,
            )
            self.model = model
            self.tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)
            self.backend = "open_clip"
            native_dim = int(getattr(model, "text_projection").shape[-1])
            if cfg.freeze_backbones:
                _freeze(model)
        except Exception as exc:  # optional dependency/checkpoint path
            if not cfg.allow_fallback:
                raise RuntimeError(
                    "CLIP ViT-B/32 could not be initialized. Install open_clip_torch "
                    "and provide the requested pretrained weights."
                ) from exc
            warnings.warn(
                f"Using CLIP fallback encoder because OpenCLIP is unavailable: {exc}",
                RuntimeWarning,
            )

        self.image_projection = nn.Linear(native_dim, output_dim)
        self.text_projection = nn.Linear(native_dim, output_dim)

    def _preprocess(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images,
            size=(self.cfg.clip_input_size, self.cfg.clip_input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = images.new_tensor(self.CLIP_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(self.CLIP_STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def encode_image(self, images: Tensor) -> Tensor:
        images = self._preprocess(images)
        if self.backend == "open_clip":
            assert self.model is not None
            features = self.model.encode_image(images)
        else:
            features = self.image_fallback(images).mean(dim=(-2, -1))
        features = F.normalize(features.float(), dim=-1)
        return self.image_projection(features)

    def encode_text(self, texts: Sequence[str], device: torch.device) -> Tensor:
        if self.backend == "open_clip":
            assert self.model is not None and self.tokenizer is not None
            tokens = self.tokenizer(list(texts)).to(device)
            features = self.model.encode_text(tokens)
        else:
            features = self.text_fallback(texts, device)
        features = F.normalize(features.float(), dim=-1)
        return self.text_projection(features)


class DINOv2Adapter(nn.Module):
    """Patch-token adapter for frozen DINOv2 visual memory.

    Role in ViSTa-Nav:
        DINOv2 extracts spatially resolved patch tokens from a historical video
        window.  Tokens retain temporal order and provide obstacle/structure
        cues for the goal-conditioned visual-memory modulation in Eqs. (7)--(8).

    Input:
        ``video`` with shape ``[B, T, 3, H, W]``.
    Output:
        patch tokens with shape ``[B, T*P, D]`` and a frame index tensor.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, output_dim: int, cfg: VisualAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.backend = "fallback"
        self.model: Optional[nn.Module] = None
        native_dim = 768  # DINOv2 ViT-B/14 patch dimension
        self.fallback = ConvImageFallback(native_dim)

        try:
            # torch.hub may use a local cache in offline reviewer environments.
            self.model = torch.hub.load("facebookresearch/dinov2", cfg.dino_model_name)
            self.backend = "torch_hub"
            native_dim = int(getattr(self.model, "embed_dim", native_dim))
            if cfg.freeze_backbones:
                _freeze(self.model)
        except Exception as exc:
            if not cfg.allow_fallback:
                raise RuntimeError(
                    "DINOv2 could not be initialized. Pre-cache the official "
                    "facebookresearch/dinov2 checkpoint or install it locally."
                ) from exc
            warnings.warn(
                f"Using DINOv2 fallback encoder because the official model is unavailable: {exc}",
                RuntimeWarning,
            )

        self.projection = nn.Linear(native_dim, output_dim)

    def _preprocess(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images,
            size=(self.cfg.dino_input_size, self.cfg.dino_input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = images.new_tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def forward(self, video: Tensor) -> Tuple[Tensor, Tensor]:
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,3,H,W], got {tuple(video.shape)}")
        batch, frames, channels, height, width = video.shape
        images = self._preprocess(video.reshape(batch * frames, channels, height, width))

        if self.backend == "torch_hub":
            assert self.model is not None
            features: Dict[str, Tensor] = self.model.forward_features(images)
            patch_tokens = features["x_norm_patchtokens"]
        else:
            feature_map = self.fallback(images)
            patch_tokens = feature_map.flatten(2).transpose(1, 2)

        patch_tokens = self.projection(patch_tokens.float())
        patch_count = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(batch, frames * patch_count, -1)
        frame_ids = torch.arange(frames, device=video.device).repeat_interleave(patch_count)
        return patch_tokens, frame_ids


class StableDiffusionVAEAdapter(nn.Module):
    """Stable-Diffusion VAE encoder adapter for latent visual foresight.

    Role in ViSTa-Nav:
        Future RGB observations are compressed into spatial latent tokens.  The
        shared predictive backbone estimates their flow velocity jointly with
        trajectory tokens, providing the auxiliary visual-foresight term in
        Eqs. (12)--(16).

    The returned latent is deterministic by default (posterior mode rather than
    posterior sampling), which reduces variance in reviewer reproduction runs.
    """

    def __init__(self, output_dim: int, cfg: VisualAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.backend = "fallback"
        self.vae: Optional[nn.Module] = None
        native_channels = 4
        self.fallback = ConvImageFallback(native_channels)

        try:
            from diffusers import AutoencoderKL  # type: ignore

            self.vae = AutoencoderKL.from_pretrained(cfg.sd_vae_model_name)
            self.backend = "diffusers"
            native_channels = int(self.vae.config.latent_channels)
            if cfg.freeze_backbones:
                _freeze(self.vae)
        except Exception as exc:
            if not cfg.allow_fallback:
                raise RuntimeError(
                    "Stable-Diffusion VAE could not be initialized. Install diffusers "
                    "and provide the requested VAE checkpoint."
                ) from exc
            warnings.warn(
                f"Using SD-VAE fallback encoder because diffusers/weights are unavailable: {exc}",
                RuntimeWarning,
            )

        self.token_projection = nn.Linear(native_channels, output_dim)

    def forward(self, future_video: Tensor) -> Tensor:
        if future_video.ndim != 5:
            raise ValueError(
                f"Expected future_video [B,T,3,H,W], got {tuple(future_video.shape)}"
            )
        batch, frames, channels, height, width = future_video.shape
        images = future_video.reshape(batch * frames, channels, height, width)
        images = images * 2.0 - 1.0  # SD-VAE convention: [0,1] -> [-1,1]

        if self.backend == "diffusers":
            assert self.vae is not None
            posterior = self.vae.encode(images).latent_dist
            latent = posterior.mode() * self.cfg.vae_scaling_factor
        else:
            latent = self.fallback(images)

        tokens = latent.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens.float())
        return tokens.view(batch, frames, tokens.shape[1], tokens.shape[2])


# -----------------------------------------------------------------------------
# Fusion and predictive world model
# -----------------------------------------------------------------------------


class GoalConditionedVisualMemory(nn.Module):
    """Residual goal modulation corresponding to manuscript Eqs. (7)--(8)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = hidden_dim ** -0.5

    def forward(self, memory_tokens: Tensor, goal_query: Tensor) -> Tensor:
        relevance = torch.sigmoid(
            torch.einsum("bnd,bd->bn", memory_tokens, goal_query) * self.scale
        ).unsqueeze(-1)
        return memory_tokens + relevance * memory_tokens


class CrossModalConditionFusion(nn.Module):
    """Fuse language, current-view, DINO memory, reliability, and mask cues."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.visual_memory = GoalConditionedVisualMemory(hidden_dim)
        self.language_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        clip_image: Tensor,
        clip_text: Tensor,
        dino_memory: Tensor,
        reliability_embedding: Tensor,
        mask_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        modulated_memory = self.visual_memory(dino_memory, clip_text)
        visual_tokens = torch.cat([clip_image.unsqueeze(1), modulated_memory], dim=1)
        language_query = clip_text.unsqueeze(1)
        attended, _ = self.language_to_visual(
            language_query,
            visual_tokens,
            visual_tokens,
            need_weights=False,
        )
        pooled_memory = self.visual_norm(modulated_memory.mean(dim=1))
        condition = self.condition_mlp(
            torch.cat(
                [
                    attended.squeeze(1),
                    pooled_memory,
                    reliability_embedding,
                    mask_embedding,
                ],
                dim=-1,
            )
        )
        return condition, modulated_memory


class SinusoidalFlowTimeEmbedding(nn.Module):
    """Continuous flow-time embedding used by the conditional flow model."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, flow_time: Tensor) -> Tensor:
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=flow_time.device, dtype=flow_time.dtype)
            / max(half - 1, 1)
        )
        phase = flow_time.reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return self.projection(embedding)


class ViSTaNav(nn.Module):
    """Visual-semantic latent predictive world model for GNSS-degraded UAV VLN.

    Reviewer-facing mapping to the manuscript:

    * ``motion_encoder`` embeds degraded ENU positions, timestamps, availability
      masks, and reliability scores (trajectory-geometric stream).
    * ``clip_adapter`` produces global image/text semantic representations.
    * ``dino_adapter`` builds patch-level historical visual memory.
    * ``sd_vae_adapter`` encodes future RGB frames for latent visual foresight.
    * ``condition_fusion`` implements language-query cross-attention over visual
      memory and explicitly includes mask/reliability embeddings.
    * ``shared_backbone`` jointly processes trajectory and visual latent tokens,
      approximating the shared DiT-style predictive diffusion backbone.
    * ``trajectory_head`` and ``visual_head`` predict conditional flow velocity
      fields for trajectory and future visual latent states, respectively.

    Notes on exact reproduction:
        The official release should pin pretrained checkpoint hashes, image
        preprocessing versions, and frame-sampling rules.  This file exposes
        those choices as explicit adapter configuration rather than hiding them
        inside the dataset loader.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_language: bool = True,
        use_visual_foresight: bool = True,
        adapter_config: Optional[VisualAdapterConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_language = use_language
        self.use_visual_foresight = use_visual_foresight
        cfg = adapter_config or VisualAdapterConfig()

        # Physical trajectory state: [east, north, up, timestamp, mask, reliability].
        self.motion_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noisy_trajectory_encoder = nn.Linear(3, hidden_dim)
        self.reliability_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.flow_time_encoder = SinusoidalFlowTimeEmbedding(hidden_dim)

        self.clip_adapter = CLIPViTB32Adapter(hidden_dim, cfg)
        self.dino_adapter = DINOv2Adapter(hidden_dim, cfg)
        self.sd_vae_adapter = StableDiffusionVAEAdapter(hidden_dim, cfg)
        self.condition_fusion = CrossModalConditionFusion(
            hidden_dim, num_heads, dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shared_backbone = nn.TransformerEncoder(encoder_layer, num_layers)
        self.trajectory_head = nn.Linear(hidden_dim, 3)
        self.visual_head = nn.Linear(hidden_dim, hidden_dim)
        self.token_type_embedding = nn.Embedding(2, hidden_dim)

    @staticmethod
    def _masked_mean(values: Tensor, valid_mask: Tensor) -> Tensor:
        weights = valid_mask.float().unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode_world_state(
        self,
        observed: Tensor,
        timestamps: Tensor,
        mask: Tensor,
        reliability: Tensor,
        instructions: Sequence[str],
        historical_video: Optional[Tensor],
        current_frame: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Encode the dual-stream world-state condition.

        ``historical_video`` is expected as ``[B,T,3,H,W]`` and ``current_frame``
        as ``[B,3,H,W]``.  If images are omitted, zero visual tokens are used so
        that legacy trajectory-only smoke tests remain valid.
        """
        x = torch.cat(
            [
                observed,
                timestamps.unsqueeze(-1),
                mask.unsqueeze(-1),
                reliability.unsqueeze(-1),
            ],
            dim=-1,
        )
        motion_tokens = self.motion_encoder(x)
        batch = observed.shape[0]
        device = observed.device

        reliability_summary = self.reliability_encoder(
            self._masked_mean(reliability.unsqueeze(-1), mask > 0.5)
        )
        mask_statistics = torch.stack(
            [mask.float().mean(dim=1), 1.0 - mask.float().mean(dim=1)], dim=-1
        )
        mask_embedding = self.mask_encoder(mask_statistics)

        if current_frame is not None:
            clip_image = self.clip_adapter.encode_image(current_frame)
        else:
            clip_image = observed.new_zeros(batch, self.hidden_dim)

        if self.use_language:
            clip_text = self.clip_adapter.encode_text(instructions, device)
        else:
            clip_text = observed.new_zeros(batch, self.hidden_dim)

        if historical_video is not None:
            dino_memory, frame_ids = self.dino_adapter(historical_video)
        else:
            dino_memory = observed.new_zeros(batch, 1, self.hidden_dim)
            frame_ids = torch.zeros(1, device=device, dtype=torch.long)

        condition, modulated_memory = self.condition_fusion(
            clip_image,
            clip_text,
            dino_memory,
            reliability_summary,
            mask_embedding,
        )
        auxiliary = {
            "clip_image": clip_image,
            "clip_text": clip_text,
            "dino_memory": dino_memory,
            "goal_modulated_memory": modulated_memory,
            "dino_frame_ids": frame_ids,
            "reliability_embedding": reliability_summary,
            "mask_embedding": mask_embedding,
        }
        return motion_tokens, condition, auxiliary

    # Backward-compatible alias used by earlier training scripts.
    def condition(
        self,
        observed: Tensor,
        timestamps: Tensor,
        mask: Tensor,
        reliability: Tensor,
        instructions: Sequence[str],
        historical_video: Optional[Tensor] = None,
        current_frame: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        motion, condition, _ = self.encode_world_state(
            observed,
            timestamps,
            mask,
            reliability,
            instructions,
            historical_video,
            current_frame,
        )
        return motion, condition

    def forward(
        self,
        noisy_trajectory: Tensor,
        flow_time: Tensor,
        observed: Tensor,
        timestamps: Tensor,
        mask: Tensor,
        reliability: Tensor,
        instructions: Sequence[str],
        padding_mask: Optional[Tensor] = None,
        historical_video: Optional[Tensor] = None,
        current_frame: Optional[Tensor] = None,
        noisy_visual_latents: Optional[Tensor] = None,
        future_video: Optional[Tensor] = None,
        return_auxiliary: bool = False,
    ):
        """Predict trajectory and visual flow velocity fields.

        Parameters
        ----------
        noisy_trajectory:
            Interpolated/noisy trajectory state ``Y_xi`` with shape ``[B,N,3]``.
        flow_time:
            Continuous conditional-flow time ``xi`` with shape ``[B]``.
        noisy_visual_latents:
            Optional interpolated visual tokens ``Z_xi`` with shape ``[B,M,D]``.
            If omitted and ``future_video`` is supplied, SD-VAE latents are used.
        future_video:
            Optional future RGB sequence ``[B,T,3,H,W]`` used to construct the
            visual-foresight stream during training.
        """
        motion_tokens, condition, auxiliary = self.encode_world_state(
            observed,
            timestamps,
            mask,
            reliability,
            instructions,
            historical_video,
            current_frame,
        )

        time_embedding = self.flow_time_encoder(flow_time).unsqueeze(1)
        trajectory_tokens = (
            self.noisy_trajectory_encoder(noisy_trajectory)
            + motion_tokens
            + condition.unsqueeze(1)
            + time_embedding
            + self.token_type_embedding.weight[0].view(1, 1, -1)
        )

        visual_shape: Optional[Tuple[int, ...]] = None
        if self.use_visual_foresight:
            if noisy_visual_latents is None and future_video is not None:
                encoded = self.sd_vae_adapter(future_video)
                visual_shape = tuple(encoded.shape)
                noisy_visual_latents = encoded.flatten(1, 2)
            elif noisy_visual_latents is not None:
                visual_shape = tuple(noisy_visual_latents.shape)

        if noisy_visual_latents is not None and self.use_visual_foresight:
            visual_tokens = (
                noisy_visual_latents
                + condition.unsqueeze(1)
                + time_embedding
                + self.token_type_embedding.weight[1].view(1, 1, -1)
            )
            all_tokens = torch.cat([trajectory_tokens, visual_tokens], dim=1)
            if padding_mask is not None:
                visual_padding = torch.zeros(
                    padding_mask.shape[0],
                    visual_tokens.shape[1],
                    dtype=torch.bool,
                    device=padding_mask.device,
                )
                full_padding_mask = torch.cat([padding_mask, visual_padding], dim=1)
            else:
                full_padding_mask = None
        else:
            all_tokens = trajectory_tokens
            full_padding_mask = padding_mask

        hidden = self.shared_backbone(
            all_tokens, src_key_padding_mask=full_padding_mask
        )
        trajectory_hidden = hidden[:, : trajectory_tokens.shape[1]]
        trajectory_velocity = self.trajectory_head(trajectory_hidden)

        visual_velocity = None
        if noisy_visual_latents is not None and self.use_visual_foresight:
            visual_hidden = hidden[:, trajectory_tokens.shape[1] :]
            visual_velocity = self.visual_head(visual_hidden)
            if visual_shape is not None and len(visual_shape) == 4:
                visual_velocity = visual_velocity.view(
                    visual_shape[0], visual_shape[1], visual_shape[2], -1
                )

        auxiliary.update(
            {
                "world_state_condition": condition,
                "trajectory_hidden": trajectory_hidden,
            }
        )

        # Preserve the original three-output interface by default.
        if return_auxiliary:
            return trajectory_velocity, visual_velocity, condition, auxiliary
        return trajectory_velocity, visual_velocity, condition
