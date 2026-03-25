"""
Train v4 entrance detection model.

Direct supervised detection on DINOv2 patch features — no JEPA.
Evaluates by ray-tracing column predictions to building polygons
and computing geographic haversine error against ground truth.
"""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model_v4 import EntranceDetectorV4
from dataset_v4 import EntranceDatasetV4
from geometry import raytrace_column_to_building, entrance_to_column, camera_hfov_deg


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def evaluate(model, val_loader, device, buildings, seq_lookup, s3_index_meta=None):
    """Evaluate model with both column-level and geographic metrics."""
    model.eval()
    total_loss = 0
    total_loss_col = 0
    total_loss_reg = 0
    total_loss_vis = 0
    n_batches = 0

    col_errors = []
    vis_correct = 0
    vis_total = 0
    geo_errors = []
    rtk_geo_errors = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['entrance_cols'],
                batch_gpu['visible_flags'],
            )

            if 'loss' in out:
                B = batch_gpu['patch_strips'].shape[0]
                total_loss += out['loss'].item() * B
                total_loss_col += out['loss_col'] * B
                total_loss_reg += out['loss_reg'] * B
                total_loss_vis += out['loss_vis'] * B
                n_batches += B

            # Per-image column error (only visible images)
            col_pred = out['col_pred'].cpu().numpy()  # (B, K)
            col_gt = batch['entrance_cols'].numpy()  # (B, K)
            vis_gt = batch['visible_flags'].numpy()  # (B, K)
            mask = batch['image_mask'].numpy()  # (B, K)
            vis_logit = out['vis_logit'].cpu().numpy()  # (B, K)

            for b in range(col_pred.shape[0]):
                for k in range(col_pred.shape[1]):
                    if not mask[b, k]:
                        continue
                    vis_pred = vis_logit[b, k] > 0
                    vis_correct += int(vis_pred == vis_gt[b, k])
                    vis_total += 1
                    if vis_gt[b, k]:
                        col_errors.append(abs(col_pred[b, k] - col_gt[b, k]))

            # Geographic evaluation via ray-tracing (per-POI multi-image fusion)
            for b in range(col_pred.shape[0]):
                bld_id = batch['building_id'][b]
                if not bld_id or bld_id not in buildings:
                    continue

                building_coords_raw = buildings[bld_id]
                if not building_coords_raw or len(building_coords_raw) < 3:
                    continue

                bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]
                gt_lat = batch['entrance_lat'][b]
                gt_lon = batch['entrance_lon'][b]
                if isinstance(gt_lat, torch.Tensor):
                    gt_lat = gt_lat.item()
                    gt_lon = gt_lon.item()

                # Ray-trace each visible image's prediction to building
                intersection_points = []
                for k in range(col_pred.shape[1]):
                    if not mask[b, k]:
                        continue
                    # Use model's visibility prediction at inference
                    if vis_logit[b, k] <= 0:
                        continue

                    pred_col = col_pred[b, k]
                    # Need camera metadata for ray-tracing
                    # We stored image_ids and can reconstruct camera params
                    # For now, use the entrance_to_column inverse via stored metadata
                    # Load image metadata from cache
                    sample_dir = Path(val_loader.dataset.cache_dir) / batch['sample_id'][b]
                    meta_path = sample_dir / 'image_meta.json'
                    try:
                        with open(meta_path) as f:
                            img_metas = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue

                    # Map slot back to original image index
                    n_images = len(img_metas)
                    indices = list(range(min(n_images, 5)))
                    if k >= len(indices):
                        continue
                    orig_k = indices[k]
                    if orig_k >= len(img_metas):
                        continue

                    m = img_metas[orig_k]
                    point = raytrace_column_to_building(
                        m['cam_lat'], m['cam_lon'],
                        m['cam_compass'], m['hfov'],
                        pred_col, bld_lnglat,
                        n_cols=16, camera_type=m['cam_type'],
                    )
                    if point is not None:
                        intersection_points.append(point)

                if intersection_points:
                    # Average intersection points
                    avg_lat = sum(p[0] for p in intersection_points) / len(intersection_points)
                    avg_lon = sum(p[1] for p in intersection_points) / len(intersection_points)
                    err = haversine_m(gt_lat, gt_lon, avg_lat, avg_lon)
                    geo_errors.append(err)

                    is_rtk = batch['has_rtk_label'][b]
                    if isinstance(is_rtk, torch.Tensor):
                        is_rtk = is_rtk.item()
                    if is_rtk:
                        rtk_geo_errors.append(err)

    n = max(n_batches, 1)
    result = {
        'loss': total_loss / n,
        'loss_col': total_loss_col / n,
        'loss_reg': total_loss_reg / n,
        'loss_vis': total_loss_vis / n,
    }

    if col_errors:
        col_arr = np.array(col_errors)
        result['col_mae'] = float(np.mean(col_arr))
        result['col_median'] = float(np.median(col_arr))

    if vis_total > 0:
        result['vis_acc'] = vis_correct / vis_total

    if geo_errors:
        geo_arr = np.array(geo_errors)
        result['geo_mae'] = float(np.mean(geo_arr))
        result['geo_median'] = float(np.median(geo_arr))
        result['geo_p90'] = float(np.percentile(geo_arr, 90))
        result['geo_n'] = len(geo_errors)

    if rtk_geo_errors:
        rtk_arr = np.array(rtk_geo_errors)
        result['rtk_geo_mae'] = float(np.mean(rtk_arr))
        result['rtk_geo_median'] = float(np.median(rtk_arr))
        result['rtk_geo_p90'] = float(np.percentile(rtk_arr, 90))
        result['rtk_n'] = len(rtk_geo_errors)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--s3-index-cache', default='')
    parser.add_argument('--output-dir', default='checkpoints_v4')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-hidden', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load buildings for geographic evaluation
    with open(args.buildings_json) as f:
        buildings = json.load(f)
    print(f"Loaded {len(buildings)} buildings")

    seq_lookup = {}
    if args.s3_index_cache:
        with open(args.s3_index_cache) as f:
            seq_lookup = json.load(f)

    train_ds = EntranceDatasetV4(args.data_dir, split='train')
    val_ds = EntranceDatasetV4(args.data_dir, split='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = EntranceDetectorV4(
        d_hidden=args.d_hidden,
        n_cols=16,
        d_pose=6,
        max_images=5,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_rtk_geo = float('inf')
    best_val_loss = float('inf')

    print(f"\nTraining Entrance Detector v4 for {args.epochs} epochs")
    print(f"  d_hidden = {args.d_hidden}, n_layers = {args.n_layers}, n_heads = {args.n_heads}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_loss_col = 0
        epoch_loss_reg = 0
        epoch_loss_vis = 0
        n_batches = 0

        t0 = time.time()
        for batch in train_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['entrance_cols'],
                batch_gpu['visible_flags'],
            )

            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += out['loss'].item()
            epoch_loss_col += out['loss_col']
            epoch_loss_reg += out['loss_reg']
            epoch_loss_vis += out['loss_vis']
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        # Evaluate every log_interval epochs or at start/end
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, buildings, seq_lookup)
        else:
            # Quick eval without geographic ray-tracing
            val_metrics = evaluate_quick(model, val_loader, device)

        record = {
            'epoch': epoch,
            'train_loss': epoch_loss / max(n_batches, 1),
            'train_loss_col': epoch_loss_col / max(n_batches, 1),
            'train_loss_reg': epoch_loss_reg / max(n_batches, 1),
            'train_loss_vis': epoch_loss_vis / max(n_batches, 1),
            'val_loss': val_metrics['loss'],
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': dt,
        }
        for k in ['col_mae', 'col_median', 'vis_acc', 'geo_mae', 'geo_median',
                   'geo_p90', 'rtk_geo_mae', 'rtk_geo_median', 'rtk_geo_p90']:
            if k in val_metrics:
                record[k] = val_metrics[k]

        history.append(record)

        if epoch % args.log_interval == 0 or epoch == 1:
            col_str = f"col_mae={val_metrics.get('col_mae', 0):.2f}" if 'col_mae' in val_metrics else ""
            vis_str = f"vis_acc={val_metrics.get('vis_acc', 0):.1%}" if 'vis_acc' in val_metrics else ""
            geo_str = ""
            if 'geo_mae' in val_metrics:
                geo_str = f"geo_mae={val_metrics['geo_mae']:.2f}m"
            rtk_str = ""
            if 'rtk_geo_mae' in val_metrics:
                rtk_str = (f" | RTK({val_metrics['rtk_n']}): "
                           f"mae={val_metrics['rtk_geo_mae']:.2f}m "
                           f"med={val_metrics['rtk_geo_median']:.2f}m "
                           f"p90={val_metrics['rtk_geo_p90']:.2f}m")

            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={record['train_loss']:.4f} "
                  f"(col={record['train_loss_col']:.4f} "
                  f"reg={record['train_loss_reg']:.4f} "
                  f"vis={record['train_loss_vis']:.4f}) | "
                  f"{col_str} {vis_str} {geo_str}"
                  f"{rtk_str} | "
                  f"lr={record['lr']:.2e} | {dt:.1f}s")

        # Save best by RTK geographic error if available, else val loss
        save_best = False
        if 'rtk_geo_mae' in val_metrics and val_metrics['rtk_geo_mae'] < best_rtk_geo:
            best_rtk_geo = val_metrics['rtk_geo_mae']
            save_best = True
        elif 'rtk_geo_mae' not in val_metrics and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_best = True

        if save_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args),
            }, out_dir / 'best_model.pt')

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, out_dir / f'checkpoint_epoch{epoch}.pt')

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'args': vars(args),
    }, out_dir / 'final_model.pt')

    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    if best_rtk_geo < float('inf'):
        print(f"  Best RTK geographic MAE: {best_rtk_geo:.2f}m")
    print(f"  Models saved to {out_dir}")


def evaluate_quick(model, val_loader, device):
    """Quick evaluation without ray-tracing (column-level metrics only)."""
    model.eval()
    total_loss = 0
    total_loss_col = 0
    total_loss_reg = 0
    total_loss_vis = 0
    n = 0
    col_errors = []
    vis_correct = 0
    vis_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['entrance_cols'],
                batch_gpu['visible_flags'],
            )

            B = batch_gpu['patch_strips'].shape[0]
            if 'loss' in out:
                total_loss += out['loss'].item() * B
                total_loss_col += out['loss_col'] * B
                total_loss_reg += out['loss_reg'] * B
                total_loss_vis += out['loss_vis'] * B
                n += B

            col_pred = out['col_pred'].cpu().numpy()
            col_gt = batch['entrance_cols'].numpy()
            vis_gt = batch['visible_flags'].numpy()
            mask = batch['image_mask'].numpy()
            vis_logit = out['vis_logit'].cpu().numpy()

            for b in range(col_pred.shape[0]):
                for k in range(col_pred.shape[1]):
                    if not mask[b, k]:
                        continue
                    vis_correct += int((vis_logit[b, k] > 0) == vis_gt[b, k])
                    vis_total += 1
                    if vis_gt[b, k]:
                        col_errors.append(abs(col_pred[b, k] - col_gt[b, k]))

    n = max(n, 1)
    result = {
        'loss': total_loss / n,
        'loss_col': total_loss_col / n,
        'loss_reg': total_loss_reg / n,
        'loss_vis': total_loss_vis / n,
    }
    if col_errors:
        col_arr = np.array(col_errors)
        result['col_mae'] = float(np.mean(col_arr))
        result['col_median'] = float(np.median(col_arr))
    if vis_total > 0:
        result['vis_acc'] = vis_correct / vis_total
    return result


if __name__ == '__main__':
    main()
