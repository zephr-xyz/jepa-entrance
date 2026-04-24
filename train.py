"""
Train JEPA entrance prediction model.
"""
import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from model import JEPAEntranceV3
from dataset import EntranceDatasetV3


def evaluate(model, val_loader, device):
    model.eval()
    total = {'loss': 0, 'pred': 0, 'sigreg': 0, 'entrance': 0, 'n': 0}
    t_errors = []
    meter_errors = []
    rtk_meter_errors = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['facade_t_cols'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['facade_feats'],
                batch_gpu['entrance_t'],
            )

            B = batch_gpu['patch_strips'].shape[0]
            total['loss'] += out['loss'].item() * B
            total['pred'] += out['loss_pred'] * B
            total['sigreg'] += out['loss_sigreg'] * B
            total['entrance'] += out['loss_entrance'] * B
            total['n'] += B

            t_err = (out['t_pred'] - batch_gpu['entrance_t']).abs()
            t_errors.extend(t_err.cpu().numpy().flatten().tolist())

            facade_lengths = batch_gpu['facade_feats'][:, 4]
            m_err = t_err.squeeze(-1) * facade_lengths
            m_err_list = m_err.cpu().numpy().flatten().tolist()
            meter_errors.extend(m_err_list)

            if 'has_rtk_label' in batch:
                for i, is_rtk in enumerate(batch['has_rtk_label']):
                    if is_rtk:
                        rtk_meter_errors.append(m_err_list[i])

    n = total['n']
    t_arr = np.array(t_errors)
    m_arr = np.array(meter_errors)

    result = {
        'loss': total['loss'] / n,
        'loss_pred': total['pred'] / n,
        'loss_sigreg': total['sigreg'] / n,
        'loss_entrance': total['entrance'] / n,
        'mae_t': float(np.mean(t_arr)),
        'median_t': float(np.median(t_arr)),
        'mae_meters': float(np.mean(m_arr)),
        'median_meters': float(np.median(m_arr)),
        'p90_meters': float(np.percentile(m_arr, 90)),
    }

    if rtk_meter_errors:
        rtk_arr = np.array(rtk_meter_errors)
        result['rtk_mae_meters'] = float(np.mean(rtk_arr))
        result['rtk_median_meters'] = float(np.median(rtk_arr))
        result['rtk_n'] = len(rtk_meter_errors)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--d-latent', type=int, default=128)
    parser.add_argument('--lambda-sigreg', type=float, default=0.05)
    parser.add_argument('--mu-entrance', type=float, default=10.0)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds = EntranceDatasetV3(args.data_dir, split='train')
    val_ds = EntranceDatasetV3(args.data_dir, split='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = JEPAEntranceV3(
        d_latent=args.d_latent,
        d_facade=32,
        lambda_sigreg=args.lambda_sigreg,
        mu_entrance=args.mu_entrance,
        max_images=5,
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
    best_val_meters = float('inf')

    print(f"\nTraining JEPA v3 for {args.epochs} epochs")
    print(f"  d_latent = {args.d_latent}")
    print(f"  λ_sigreg = {args.lambda_sigreg}")
    print(f"  μ_entrance = {args.mu_entrance}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_pred = 0
        epoch_sig = 0
        epoch_ent = 0
        n_batches = 0

        t0 = time.time()
        for batch in train_loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            out = model(
                batch_gpu['patch_strips'],
                batch_gpu['facade_t_cols'],
                batch_gpu['camera_poses'],
                batch_gpu['image_mask'],
                batch_gpu['facade_feats'],
                batch_gpu['entrance_t'],
            )

            optimizer.zero_grad()
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += out['loss'].item()
            epoch_pred += out['loss_pred']
            epoch_sig += out['loss_sigreg']
            epoch_ent += out['loss_entrance']
            n_batches += 1

        scheduler.step()
        dt = time.time() - t0

        val_metrics = evaluate(model, val_loader, device)

        record = {
            'epoch': epoch,
            'train_loss': epoch_loss / n_batches,
            'train_pred': epoch_pred / n_batches,
            'train_sigreg': epoch_sig / n_batches,
            'train_entrance': epoch_ent / n_batches,
            'val_loss': val_metrics['loss'],
            'val_mae_t': val_metrics['mae_t'],
            'val_mae_meters': val_metrics['mae_meters'],
            'val_median_meters': val_metrics['median_meters'],
            'val_p90_meters': val_metrics['p90_meters'],
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': dt,
        }
        if 'rtk_mae_meters' in val_metrics:
            record['rtk_mae_meters'] = val_metrics['rtk_mae_meters']
            record['rtk_median_meters'] = val_metrics['rtk_median_meters']

        history.append(record)

        if epoch % args.log_interval == 0 or epoch == 1:
            rtk_str = ""
            if 'rtk_mae_meters' in val_metrics:
                rtk_str = (f" | RTK({val_metrics['rtk_n']}): "
                           f"mae={val_metrics['rtk_mae_meters']:.2f}m "
                           f"med={val_metrics['rtk_median_meters']:.2f}m")
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={record['train_loss']:.4f} "
                  f"(pred={record['train_pred']:.4f} "
                  f"sig={record['train_sigreg']:.4f} "
                  f"ent={record['train_entrance']:.6f}) | "
                  f"val_mae={val_metrics['mae_meters']:.2f}m "
                  f"(med={val_metrics['median_meters']:.2f}m "
                  f"p90={val_metrics['p90_meters']:.2f}m)"
                  f"{rtk_str} | "
                  f"lr={record['lr']:.2e} | {dt:.1f}s")

        if val_metrics['mae_meters'] < best_val_meters:
            best_val_meters = val_metrics['mae_meters']
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
    print(f"  Best val MAE: {best_val_meters:.2f}m")
    print(f"  Models saved to {out_dir}")

    # Baseline comparison
    print(f"\n--- Baseline vs JEPA v3 ---")
    val_manifest = json.load(open(Path(args.data_dir) / 'val_manifest.json'))
    baseline_errors = [abs(e['entrance_t'] - 0.5) * e['facade_length_m'] for e in val_manifest]
    baseline_errors = np.array(baseline_errors)
    print(f"  Baseline MAE: {np.mean(baseline_errors):.2f}m "
          f"(med={np.median(baseline_errors):.2f}m p90={np.percentile(baseline_errors, 90):.2f}m)")
    print(f"  JEPA v3 MAE:  {best_val_meters:.2f}m")
    if np.mean(baseline_errors) > 0:
        improvement = (1 - best_val_meters / np.mean(baseline_errors)) * 100
        print(f"  Improvement:  {improvement:.1f}%")


if __name__ == '__main__':
    main()
