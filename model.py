"""
JEPA model for building entrance prediction.

Adapted from LeWorldModel (Maes et al. 2026) — a stable end-to-end JEPA
trained with only two losses: MSE prediction + SIGReg anti-collapse.

Architecture:
  Context Encoder: Fuses Mapillary visual features (DINOv2 CLS, patch
    embeddings, caption embedding, keypoint stats) into a latent z_visual.
  Target Encoder: Encodes facade geometry + true entrance position into z_geo.
  Predictor: Maps z_visual → ẑ_geo, conditioned on facade geometry via AdaLN.
  Entrance Head: Decodes ẑ_geo → predicted entrance parameter t ∈ [0, 1]
    (fraction along the facade edge).

Training loss:
  L = MSE(ẑ_geo, z_geo) + λ·SIGReg(Z) + μ·MSE(t_pred, t_true)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SIGReg: Sketched Isotropic Gaussian Regularizer (from LeWM)
# ---------------------------------------------------------------------------

def sigreg(z: torch.Tensor, n_projections: int = 512) -> torch.Tensor:
    """SIGReg anti-collapse regularizer.

    Projects embeddings onto random unit directions and penalizes
    deviation from a standard normal distribution using Epps-Pulley
    test statistic. By Cramér-Wold theorem, matching all 1D marginals
    matches the full joint distribution.

    Args:
        z: (B, D) batch of latent embeddings
        n_projections: number of random projection directions M
    Returns:
        scalar loss (lower = more Gaussian)
    """
    B, D = z.shape
    if B < 4:
        return torch.tensor(0.0, device=z.device)

    # Standardize per-dimension
    z_std = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-8)

    # Random unit-norm projection directions
    directions = torch.randn(D, n_projections, device=z.device)
    directions = F.normalize(directions, dim=0)

    # Project: (B, M)
    projections = z_std @ directions

    # Epps-Pulley test statistic per projection
    # Simplified: penalize skewness and excess kurtosis of each projection
    # (approximation of the full EP test for efficiency)
    mean_p = projections.mean(dim=0)
    var_p = projections.var(dim=0)
    std_p = var_p.sqrt() + 1e-8

    centered = (projections - mean_p.unsqueeze(0)) / std_p.unsqueeze(0)
    skew = (centered ** 3).mean(dim=0)
    kurt = (centered ** 4).mean(dim=0) - 3.0  # excess kurtosis

    # Combined normality penalty
    loss = (skew ** 2).mean() + (kurt ** 2).mean() + ((var_p - 1.0) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------
# Adaptive Layer Normalization (from LeWM predictor)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on an external signal.
    Used to inject facade geometry into the predictor (analogous to
    how LeWM injects actions via AdaLN).
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        # Zero-init so conditioning has progressive effect (LeWM trick)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


# ---------------------------------------------------------------------------
# Context Encoder — fuses multi-modal Mapillary features
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """Encodes Mapillary visual features into a latent representation.

    Inputs:
        cls_emb:     (B, 1024) DINOv2-large CLS token
        caption_emb: (B, 768)  Gemma caption embedding
        kp_stats:    (B, 16)   Keypoint statistics (count, spatial distribution)
        compass:     (B, 1)    Camera compass angle (normalized)

    Output:
        z_visual:    (B, D)    Latent visual embedding
    """

    def __init__(self, d_latent: int = 256, d_hidden: int = 512):
        super().__init__()
        # Project each modality to common dimension
        self.cls_proj = nn.Linear(1024, d_hidden)
        self.caption_proj = nn.Linear(768, d_hidden)
        self.kp_proj = nn.Linear(16, d_hidden)
        self.compass_proj = nn.Linear(1, d_hidden)

        # Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)

        # Projection head (MLP + BatchNorm, per LeWM)
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_latent),
            nn.BatchNorm1d(d_latent),
        )

    def forward(self, cls_emb, caption_emb, kp_stats, compass):
        B = cls_emb.shape[0]

        # Project each modality to tokens
        tokens = torch.stack([
            self.cls_proj(cls_emb),
            self.caption_proj(caption_emb),
            self.kp_proj(kp_stats),
            self.compass_proj(compass),
        ], dim=1)  # (B, 4, H)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 5, H)

        # Fuse
        out = self.fusion(tokens)
        z = out[:, 0]  # CLS output

        return self.projector(z)  # (B, D)


# ---------------------------------------------------------------------------
# Target Encoder — encodes facade geometry + entrance position
# ---------------------------------------------------------------------------

class TargetEncoder(nn.Module):
    """Encodes building facade geometry and entrance position.

    Inputs:
        facade_feats: (B, F) facade features:
            - facade edge vertices (start_x, start_y, end_x, end_y in local meters)
            - facade length, bearing
            - building centroid offset
            - road class encoding
        entrance_t:   (B, 1) entrance position as fraction along facade [0, 1]

    Output:
        z_geo: (B, D) latent geospatial embedding
    """

    def __init__(self, d_facade: int = 32, d_latent: int = 256, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_facade + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_latent),
            nn.BatchNorm1d(d_latent),
        )

    def forward(self, facade_feats, entrance_t):
        x = torch.cat([facade_feats, entrance_t], dim=-1)
        h = self.net(x)
        return self.projector(h)


# ---------------------------------------------------------------------------
# Predictor — maps z_visual → ẑ_geo, conditioned on facade via AdaLN
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """Predicts target geospatial embedding from visual context,
    conditioned on facade geometry via AdaLN (analogous to LeWM's
    action conditioning).

    Input:
        z_visual:     (B, D) from context encoder
        facade_cond:  (B, F) facade geometry features (conditioning signal)
    Output:
        z_geo_pred:   (B, D) predicted target embedding
    """

    def __init__(self, d_latent: int = 256, d_facade: int = 32,
                 n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_latent, d_latent)

        self.layers = nn.ModuleList()
        self.adaln_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=d_latent, nhead=n_heads,
                dim_feedforward=d_latent * 4,
                dropout=dropout, batch_first=True, activation='gelu'
            ))
            self.adaln_layers.append(AdaLN(d_latent, d_facade))

        self.projector = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.BatchNorm1d(d_latent),
        )

    def forward(self, z_visual, facade_cond):
        x = self.input_proj(z_visual).unsqueeze(1)  # (B, 1, D)

        for layer, adaln in zip(self.layers, self.adaln_layers):
            x = layer(x)
            x = adaln(x.squeeze(1), facade_cond).unsqueeze(1)

        return self.projector(x.squeeze(1))


# ---------------------------------------------------------------------------
# Entrance Decode Head
# ---------------------------------------------------------------------------

class EntranceHead(nn.Module):
    """Decodes predicted geospatial embedding to entrance position t ∈ [0,1]."""

    def __init__(self, d_latent: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_geo):
        return self.net(z_geo)


# ---------------------------------------------------------------------------
# Full JEPA Entrance Model
# ---------------------------------------------------------------------------

class JEPAEntrance(nn.Module):
    """Full JEPA model for entrance prediction.

    Follows LeWM's two-loss training:
      L = MSE(ẑ_geo, z_geo) + λ·SIGReg(Z) + μ·MSE(t_pred, t_true)
    """

    def __init__(self, d_latent: int = 256, d_facade: int = 32,
                 lambda_sigreg: float = 0.1, mu_entrance: float = 1.0):
        super().__init__()
        self.context_encoder = ContextEncoder(d_latent=d_latent)
        self.target_encoder = TargetEncoder(d_facade=d_facade, d_latent=d_latent)
        self.predictor = Predictor(d_latent=d_latent, d_facade=d_facade)
        self.entrance_head = EntranceHead(d_latent=d_latent)

        self.lambda_sigreg = lambda_sigreg
        self.mu_entrance = mu_entrance

    def forward(self, cls_emb, caption_emb, kp_stats, compass,
                facade_feats, entrance_t):
        """Full forward pass for training.

        Returns dict with losses and predictions.
        """
        # Encode visual context
        z_visual = self.context_encoder(cls_emb, caption_emb, kp_stats, compass)

        # Encode target (facade + true entrance)
        z_geo = self.target_encoder(facade_feats, entrance_t)

        # Predict target from visual context, conditioned on facade
        z_geo_pred = self.predictor(z_visual, facade_feats)

        # Decode entrance position
        t_pred = self.entrance_head(z_geo_pred)

        # --- Losses (LeWM style) ---
        # 1. Prediction loss: MSE in latent space
        loss_pred = F.mse_loss(z_geo_pred, z_geo)

        # 2. SIGReg on both embedding spaces
        loss_sigreg = sigreg(z_visual) + sigreg(z_geo)

        # 3. Entrance position loss (auxiliary decode supervision)
        loss_entrance = F.mse_loss(t_pred, entrance_t)

        # Total loss (LeWM Eq. 3 + entrance decode)
        loss = loss_pred + self.lambda_sigreg * loss_sigreg + self.mu_entrance * loss_entrance

        return {
            'loss': loss,
            'loss_pred': loss_pred.item(),
            'loss_sigreg': loss_sigreg.item(),
            'loss_entrance': loss_entrance.item(),
            'z_visual': z_visual,
            'z_geo': z_geo,
            'z_geo_pred': z_geo_pred,
            't_pred': t_pred,
        }

    @torch.no_grad()
    def predict_entrance(self, cls_emb, caption_emb, kp_stats, compass,
                         facade_feats):
        """Inference: predict entrance position from visual features + facade."""
        z_visual = self.context_encoder(cls_emb, caption_emb, kp_stats, compass)
        z_geo_pred = self.predictor(z_visual, facade_feats)
        t_pred = self.entrance_head(z_geo_pred)
        return t_pred
