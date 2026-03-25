"""
JEPA v3 model for building entrance prediction.

Key changes from v2:
  - Uses DINOv2 patch embeddings (16x16 grid → 16 horizontal strip after vertical pooling)
  - Camera pose features encode spatial relationship between camera and facade
  - Per-patch facade_t positional encoding (where each patch column looks on the facade)
  - Multi-image support: fuses K viewpoints per POI via cross-attention
  - Retains LeWM training paradigm: MSE prediction + SIGReg + entrance decode

Architecture:
  Context Encoder: Patch strip tokens (16 per image × K images) with facade_t
    positional encoding + camera pose conditioning → z_visual
  Target Encoder: Facade geometry + true entrance_t → z_geo
  Predictor: z_visual → ẑ_geo conditioned on facade via AdaLN
  Entrance Head: ẑ_geo → t_pred ∈ [0, 1]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SIGReg (same as v2)
# ---------------------------------------------------------------------------

def sigreg(z: torch.Tensor, n_projections: int = 512) -> torch.Tensor:
    B, D = z.shape
    if B < 4:
        return torch.tensor(0.0, device=z.device)

    z_std = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-8)
    directions = torch.randn(D, n_projections, device=z.device)
    directions = F.normalize(directions, dim=0)
    projections = z_std @ directions

    mean_p = projections.mean(dim=0)
    var_p = projections.var(dim=0)
    skew = ((projections - mean_p) ** 3).mean(dim=0) / (var_p + 1e-8) ** 1.5
    kurt = ((projections - mean_p) ** 4).mean(dim=0) / (var_p + 1e-8) ** 2 - 3.0

    loss_mean = mean_p.pow(2).mean()
    loss_var = (var_p - 1).pow(2).mean()
    loss_skew = skew.pow(2).mean()
    loss_kurt = kurt.pow(2).mean()

    return loss_mean + loss_var + 0.5 * loss_skew + 0.25 * loss_kurt


# ---------------------------------------------------------------------------
# AdaLN (same as v2)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Facade-t Positional Encoding
# ---------------------------------------------------------------------------

class FacadeTPositionalEncoding(nn.Module):
    """Encode per-patch facade_t values into sinusoidal positional embeddings."""

    def __init__(self, d_model: int, max_freq: int = 64):
        super().__init__()
        self.d_model = d_model
        freqs = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.register_buffer('freqs', freqs)

    def forward(self, facade_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            facade_t: (B, N) values in [0, 1] or [-1, 2] (can be outside facade)
        Returns:
            (B, N, d_model) positional embeddings
        """
        # Scale to reasonable range for sinusoidal encoding
        t = facade_t.unsqueeze(-1) * math.pi  # (B, N, 1)
        pe = torch.zeros(*facade_t.shape, self.d_model, device=facade_t.device)
        pe[..., 0::2] = torch.sin(t * self.freqs)
        pe[..., 1::2] = torch.cos(t * self.freqs)
        return pe


# ---------------------------------------------------------------------------
# Context Encoder v3 — patch-level with camera pose
# ---------------------------------------------------------------------------

class ContextEncoderV3(nn.Module):
    """Encodes spatially-registered DINOv2 patch features from multiple viewpoints.

    Per image:
      - Horizontal patch strip: (16, 1024) — vertical average of 16x16 patch grid
      - Per-column facade_t: (16,) — where each column looks on the facade
      - Camera pose: (8,) — spatial relationship to facade

    Fuses K images via transformer with facade_t positional encoding.

    Inputs:
        patch_strips:  (B, K, 16, 1024)  DINOv2 horizontal strip per image
        facade_t_cols: (B, K, 16)         facade_t for each patch column
        camera_poses:  (B, K, 8)          camera-facade geometry per image
        image_mask:    (B, K)             bool mask (True = valid image)

    Output:
        z_visual: (B, D) latent visual embedding
    """

    def __init__(self, d_latent: int = 128, d_hidden: int = 256,
                 n_cols: int = 16, max_images: int = 5):
        super().__init__()
        self.n_cols = n_cols
        self.max_images = max_images
        self.d_hidden = d_hidden

        # Project patch features
        self.patch_proj = nn.Linear(1024, d_hidden)

        # Facade_t positional encoding
        self.facade_t_pe = FacadeTPositionalEncoding(d_hidden)

        # Camera pose projection (injected as a per-image token)
        self.pose_proj = nn.Sequential(
            nn.Linear(8, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Image-level token (like CLS, one per image to aggregate)
        self.image_token = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)

        # Self-attention across all patch tokens from all images
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Global CLS token for final aggregation
        self.global_cls = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)

        # Output projection
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_latent),
            nn.BatchNorm1d(d_latent),
        )

    def forward(self, patch_strips, facade_t_cols, camera_poses, image_mask):
        B = patch_strips.shape[0]
        K = patch_strips.shape[1]

        all_tokens = []
        all_masks = []

        for k in range(K):
            # Patch features: (B, 16, 1024) → (B, 16, H)
            patches = self.patch_proj(patch_strips[:, k])  # (B, 16, H)

            # Add facade_t positional encoding
            ft_pe = self.facade_t_pe(facade_t_cols[:, k])  # (B, 16, H)
            patches = patches + ft_pe

            # Camera pose token: (B, 8) → (B, 1, H)
            pose_tok = self.pose_proj(camera_poses[:, k]).unsqueeze(1)  # (B, 1, H)

            # Image aggregation token
            img_tok = self.image_token.expand(B, -1, -1)  # (B, 1, H)

            # Combine: [img_tok, pose_tok, patch_0, ..., patch_15]
            img_tokens = torch.cat([img_tok, pose_tok, patches], dim=1)  # (B, 18, H)

            # Mask: expand image_mask to token level
            # image_mask[:, k] is True if image k is valid
            valid = image_mask[:, k]  # (B,)
            token_mask = ~valid.unsqueeze(1).expand(-1, 18)  # (B, 18) — True means IGNORE in PyTorch

            all_tokens.append(img_tokens)
            all_masks.append(token_mask)

        # Prepend global CLS token
        global_cls = self.global_cls.expand(B, -1, -1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=patch_strips.device)

        # Concatenate across all images: (B, 1 + K*18, H)
        tokens = torch.cat([global_cls] + all_tokens, dim=1)
        mask = torch.cat([cls_mask] + all_masks, dim=1)

        # Self-attention
        out = self.transformer(tokens, src_key_padding_mask=mask)

        # Extract global CLS
        z = out[:, 0]  # (B, H)

        return self.projector(z)  # (B, D)


# ---------------------------------------------------------------------------
# Target Encoder (same as v2)
# ---------------------------------------------------------------------------

class TargetEncoder(nn.Module):
    def __init__(self, d_latent: int = 128, d_facade: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_facade + 1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, d_latent),
            nn.BatchNorm1d(d_latent),
        )

    def forward(self, facade_feats, entrance_t):
        x = torch.cat([facade_feats, entrance_t], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Predictor with AdaLN (same structure as v2)
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    def __init__(self, d_latent: int = 128, d_facade: int = 32, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'adaln': AdaLN(d_latent, d_facade),
                'ffn': nn.Sequential(
                    nn.Linear(d_latent, d_latent * 4),
                    nn.GELU(),
                    nn.Linear(d_latent * 4, d_latent),
                ),
            }))

    def forward(self, z_visual, facade_feats):
        x = z_visual
        for layer in self.layers:
            x = layer['adaln'](x, facade_feats)
            x = x + layer['ffn'](x)
        return x


# ---------------------------------------------------------------------------
# Entrance Head (same as v2)
# ---------------------------------------------------------------------------

class EntranceHead(nn.Module):
    def __init__(self, d_latent: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Full JEPA v3 Model
# ---------------------------------------------------------------------------

class JEPAEntranceV3(nn.Module):
    def __init__(self, d_latent: int = 128, d_facade: int = 32,
                 lambda_sigreg: float = 0.05, mu_entrance: float = 10.0,
                 max_images: int = 5):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.mu_entrance = mu_entrance

        self.context_encoder = ContextEncoderV3(
            d_latent=d_latent, d_hidden=d_latent * 2,
            max_images=max_images,
        )
        self.target_encoder = TargetEncoder(d_latent=d_latent, d_facade=d_facade)
        self.predictor = Predictor(d_latent=d_latent, d_facade=d_facade, n_layers=4)
        self.entrance_head = EntranceHead(d_latent=d_latent)

    def forward(self, patch_strips, facade_t_cols, camera_poses, image_mask,
                facade_feats, entrance_t):
        # Context: visual features → z_visual
        z_visual = self.context_encoder(
            patch_strips, facade_t_cols, camera_poses, image_mask
        )

        # Target: facade + entrance → z_geo
        z_geo = self.target_encoder(facade_feats, entrance_t)

        # Predict: z_visual → ẑ_geo
        z_geo_pred = self.predictor(z_visual, facade_feats)

        # Entrance decode
        t_pred = self.entrance_head(z_geo_pred)

        # Losses
        loss_pred = F.mse_loss(z_geo_pred, z_geo)
        loss_sigreg = sigreg(z_visual) + sigreg(z_geo)
        loss_entrance = F.mse_loss(t_pred, entrance_t)

        loss = (loss_pred
                + self.lambda_sigreg * loss_sigreg
                + self.mu_entrance * loss_entrance)

        return {
            'loss': loss,
            'loss_pred': loss_pred.item(),
            'loss_sigreg': loss_sigreg.item(),
            'loss_entrance': loss_entrance.item(),
            't_pred': t_pred.detach(),
            'z_visual': z_visual.detach(),
            'z_geo': z_geo.detach(),
        }
