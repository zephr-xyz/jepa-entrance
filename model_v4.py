"""
v4 Entrance Detection Model.

Detects which image column contains the building entrance, then uses
ray-tracing geometry to map from image space to geographic coordinates.

No JEPA — direct supervised detection on DINOv2 patch features.

Architecture:
  Per image:
    Patch strip (16×1024) + column positional encoding + camera pose
    → Transformer self-attention
    → Per-column entrance logits (16-way softmax)
    → Soft-argmax for continuous column prediction
    → Visibility score (is entrance in this image?)

  Multi-image fusion happens at inference via ray-tracing geometry,
  not inside the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntranceDetectorV4(nn.Module):
    def __init__(self, d_hidden=256, n_cols=16, d_pose=6, max_images=5,
                 n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_cols = n_cols
        self.max_images = max_images
        self.d_hidden = d_hidden

        # Project DINOv2 patch features
        self.patch_proj = nn.Linear(1024, d_hidden)

        # Learnable column positional encoding
        self.col_pos = nn.Parameter(torch.randn(1, n_cols, d_hidden) * 0.02)

        # Camera pose projection (added to all tokens)
        self.pose_proj = nn.Sequential(
            nn.Linear(d_pose, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # Per-image transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-column entrance logit
        self.col_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Visibility head (is entrance visible in this image?)
        self.vis_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Column indices for soft-argmax
        self.register_buffer('col_indices',
                             torch.arange(n_cols, dtype=torch.float32))

    def forward_single_image(self, patch_strip, camera_pose):
        """Process one image.

        Args:
            patch_strip: (B, 16, 1024)
            camera_pose: (B, 6)

        Returns:
            col_logits: (B, 16) raw logits per column
            vis_logit: (B,) visibility logit
        """
        B = patch_strip.shape[0]

        # Project patches + add column positional encoding
        tokens = self.patch_proj(patch_strip) + self.col_pos  # (B, 16, H)

        # Add camera pose as global conditioning
        pose_emb = self.pose_proj(camera_pose).unsqueeze(1)  # (B, 1, H)
        tokens = tokens + pose_emb

        # Self-attention
        tokens = self.transformer(tokens)  # (B, 16, H)

        # Per-column entrance logits
        col_logits = self.col_head(tokens).squeeze(-1)  # (B, 16)

        # Visibility: mean-pool then classify
        vis_logit = self.vis_head(tokens.mean(dim=1)).squeeze(-1)  # (B,)

        return col_logits, vis_logit

    def forward(self, patch_strips, camera_poses, image_mask,
                entrance_cols=None, visible_flags=None):
        """Process K images per POI.

        Args:
            patch_strips: (B, K, 16, 1024)
            camera_poses: (B, K, 6)
            image_mask: (B, K) bool — True if image slot is valid
            entrance_cols: (B, K) float — ground truth column (for training)
            visible_flags: (B, K) bool — entrance visible in image (for training)

        Returns dict with:
            col_logits: (B, K, 16) per-image column logits
            col_pred: (B, K) predicted continuous column (soft-argmax)
            vis_logit: (B, K) visibility logits
            loss: scalar (if training targets provided)
        """
        B, K = patch_strips.shape[:2]

        all_col_logits = []
        all_vis_logits = []

        for k in range(K):
            col_logits, vis_logit = self.forward_single_image(
                patch_strips[:, k], camera_poses[:, k]
            )
            all_col_logits.append(col_logits)
            all_vis_logits.append(vis_logit)

        col_logits = torch.stack(all_col_logits, dim=1)  # (B, K, 16)
        vis_logits = torch.stack(all_vis_logits, dim=1)  # (B, K)

        # Soft-argmax for continuous column prediction
        col_probs = F.softmax(col_logits, dim=-1)  # (B, K, 16)
        col_pred = (col_probs * self.col_indices).sum(dim=-1)  # (B, K)

        result = {
            'col_logits': col_logits,
            'col_pred': col_pred,
            'col_probs': col_probs,
            'vis_logit': vis_logits,
        }

        # Compute loss if training targets provided
        if entrance_cols is not None and visible_flags is not None:
            loss_col = torch.tensor(0.0, device=patch_strips.device)
            loss_reg = torch.tensor(0.0, device=patch_strips.device)
            loss_vis = torch.tensor(0.0, device=patch_strips.device)
            n_vis = 0
            n_valid = 0

            for k in range(K):
                valid = image_mask[:, k]  # (B,)
                if not valid.any():
                    continue

                # Visibility loss (all valid images)
                vis_gt = visible_flags[:, k][valid].float()
                loss_vis = loss_vis + F.binary_cross_entropy_with_logits(
                    vis_logits[:, k][valid], vis_gt
                )
                n_valid += 1

                # Column loss (only visible images)
                vis_mask = valid & visible_flags[:, k]
                if not vis_mask.any():
                    continue

                # Cross-entropy on discretized column
                col_gt_int = entrance_cols[:, k][vis_mask].long().clamp(0, self.n_cols - 1)
                loss_col = loss_col + F.cross_entropy(
                    col_logits[:, k][vis_mask], col_gt_int
                )

                # Regression loss on continuous column (soft-argmax)
                col_gt_cont = entrance_cols[:, k][vis_mask].float()
                loss_reg = loss_reg + F.smooth_l1_loss(
                    col_pred[:, k][vis_mask], col_gt_cont
                )

                n_vis += 1

            n_vis = max(n_vis, 1)
            n_valid = max(n_valid, 1)

            loss = (loss_col / n_vis
                    + 0.5 * loss_reg / n_vis
                    + 0.3 * loss_vis / n_valid)

            result['loss'] = loss
            result['loss_col'] = (loss_col / n_vis).item()
            result['loss_reg'] = (loss_reg / n_vis).item()
            result['loss_vis'] = (loss_vis / n_valid).item()

        return result
