"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DRONE SNN TRAINING — FINE-TUNING FROM N-MNIST                               ║
║  "Transfer learning for obstacle detection"                                  ║
║                                                                              ║
║  Strategy:                                                                   ║
║  1. Load N-MNIST best.pt (97.11% accuracy)                                   ║
║  2. Freeze encoder, train detection head only                                ║
║  3. Unfreeze all, fine-tune end-to-end                                       ║
║  4. Save best model for deployment                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import argparse
import time
import json
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from model_drone import DroneSNN, DetectionLoss, get_model_summary
from data_drone import MVSECDataModule


# ──────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class DroneSNNTrainer:
    """
    Fine-tuning trainer for DroneSNN.
    Supports:
    - Loading N-MNIST pre-trained weights
    - Two-phase training (head-only → full fine-tuning)
    - Checkpointing and resume
    """

    def __init__(
        self,
        model: DroneSNN,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        pretrained_path: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-4,  # Lower LR for fine-tuning
            weight_decay=1e-4
        )

        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        self.criterion = DetectionLoss(
            pos_weight=1.0,
            depth_weight=2.0,
            conf_weight=1.0,
            spike_weight=0.01  # 0.01 — sparse firing without killing signal during full fine-tuning
        )

        # Logging
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / 'results' / 'logs_drone'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent / 'results' / 'checkpoints_drone'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))
        self.history = {'train': [], 'val': []}
        self.best_val_loss = float('inf')

        # Load pre-trained weights if specified
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        print(f"\n🔥 Training on: {self.device.upper()}")
        print(get_model_summary(model))

    def _load_pretrained(self, path: str):
        """
        Load N-MNIST pre-trained weights.
        Adapts weights to new architecture where needed.
        """
        print(f"\n📥 Loading pre-trained weights from: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        pretrained_state = checkpoint['model_state_dict']

        # Load compatible weights
        model_state = self.model.state_dict()

        loaded = 0
        skipped = 0
        adapted = 0

        for key in model_state:
            if key in pretrained_state:
                if model_state[key].shape == pretrained_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded += 1
                else:
                    # Shape mismatch — skip or adapt
                    print(f"   ⚠️  Shape mismatch: {key}")
                    print(f"       Model: {model_state[key].shape}, Pretrained: {pretrained_state[key].shape}")
                    skipped += 1
            else:
                # New layer (stem, detection head)
                adapted += 1

        self.model.load_state_dict(model_state, strict=False)

        print(f"   ✅ Loaded: {loaded} layers")
        print(f"   ⏭️  Skipped: {skipped} layers (shape mismatch)")
        print(f"   🆕 New: {adapted} layers (random init)")

    def freeze_encoder(self):
        """Freeze encoder layers, train only detection head."""
        print("\n❄️  Freezing encoder, training detection head only...")

        for name, param in self.model.named_parameters():
            if 'detection_head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Verify
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        print("\n🔥 Unfreezing all layers for full fine-tuning...")

        for param in self.model.parameters():
            param.requires_grad = True

    def train_epoch(self, epoch: int) -> Tuple[float, Dict, Dict]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        all_components = {'position': 0, 'depth': 0, 'confidence': 0, 'spike': 0}
        all_metrics = {'depth_mae': 0, 'depth_rmse': 0, 'pos_mae': 0, 'rel_depth_err': 0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [TRAIN]')

        for events, targets, _ in pbar:
            events = events.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(events)

            detection = output['detection']
            spike_rate = output.get('spike_rate', None)

            # Compute loss (includes spike regularization)
            loss, components = self.criterion(detection, targets, spike_rate)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # ── Accuracy metrics ─────────────────────────────────────────────
            # Depth MAE
            depth_mae = F.l1_loss(detection[:, 2, :, :], targets[:, 2, :, :]).item()
            # Depth RMSE
            depth_rmse = torch.sqrt(F.mse_loss(detection[:, 2, :, :], targets[:, 2, :, :])).item()
            # Position MAE
            pos_mae = F.l1_loss(detection[:, :2, :, :], targets[:, :2, :, :]).item()
            # Relative depth error (%)
            with torch.no_grad():
                target_depth_abs = targets[:, 2, :, :].abs()
                # Use mean depth as floor — prevents division by near-zero
                floor = target_depth_abs.mean().clamp(min=0.01)
                rel_err = (detection[:, 2, :, :] - targets[:, 2, :, :]).abs() / (target_depth_abs + floor)
                rel_depth_err = rel_err.mean().item()

            # Accumulate
            total_loss += loss.item()
            for key in all_components:
                all_components[key] += components.get(key, 0)
            for key in all_metrics:
                all_metrics[key] += locals()[key]
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'depth': f'{components["depth"]:.4f}',
                'depth_mae': f'{depth_mae:.4f}',
                'spikes': f'{spike_rate.item() if spike_rate is not None else 0:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in all_components.items()}
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        return avg_loss, avg_components, avg_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict, Dict]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        all_components = {'position': 0, 'depth': 0, 'confidence': 0, 'spike': 0}
        all_metrics = {'depth_mae': 0, 'depth_rmse': 0, 'pos_mae': 0, 'rel_depth_err': 0}
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [VAL]  ')

        for events, targets, _ in pbar:
            events = events.to(self.device)
            targets = targets.to(self.device)

            output = self.model(events)
            detection = output['detection']
            spike_rate = output.get('spike_rate', None)

            loss, components = self.criterion(detection, targets, spike_rate)

            # ── Accuracy metrics ─────────────────────────────────────────────
            depth_mae = F.l1_loss(detection[:, 2, :, :], targets[:, 2, :, :]).item()
            depth_rmse = torch.sqrt(F.mse_loss(detection[:, 2, :, :], targets[:, 2, :, :])).item()
            pos_mae = F.l1_loss(detection[:, :2, :, :], targets[:, :2, :, :]).item()
            with torch.no_grad():
                target_depth_abs = targets[:, 2, :, :].abs()
                floor = target_depth_abs.mean().clamp(min=0.01)
                rel_err = (detection[:, 2, :, :] - targets[:, 2, :, :]).abs() / (target_depth_abs + floor)
                rel_depth_err = rel_err.mean().item()

            total_loss += loss.item()
            for key in all_components:
                all_components[key] += components.get(key, 0)
            for key in all_metrics:
                all_metrics[key] += locals()[key]
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'depth_mae': f'{depth_mae:.4f}',
                'spikes': f'{spike_rate.item() if spike_rate is not None else 0:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in all_components.items()}
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}

        return avg_loss, avg_components, avg_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"   💾 New best model saved! (Loss: {self.best_val_loss:.4f})")

    def train(
        self,
        num_epochs: int = 100,
        head_only_epochs: int = 10,
        resume_from: Optional[str] = None
    ):
        """
        Full training loop with two-phase approach.

        Args:
            num_epochs: Total epochs
            head_only_epochs: Epochs with frozen encoder
            resume_from: Path to checkpoint
        """
        start_epoch = 0

        # Resume
        if resume_from:
            print(f"\n📥 Resuming from: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"   Resumed from epoch {start_epoch}")

        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"   Phase 1: Head-only (epochs {start_epoch}-{head_only_epochs})")
        print(f"   Phase 2: Full fine-tuning (epochs {head_only_epochs}-{num_epochs})\n")

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()

            # Phase transition: unfreeze after head_only_epochs
            if epoch == head_only_epochs:
                self.unfreeze_all()
                # Rebuild optimizer with clean state - encoder needs fresh Adam buffers
                old_lr = self.optimizer.param_groups[0]['lr']
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=old_lr * 0.5,  # Gentle LR: 5e-5 from 1e-4
                    weight_decay=1e-4
                )
                # Reset scheduler
                if self.scheduler:
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=num_epochs - head_only_epochs,
                        eta_min=1e-6
                    )

            # Phase 1: freeze encoder
            if epoch < head_only_epochs:
                self.freeze_encoder()

            # Train
            train_loss, train_components, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_components, val_metrics = self.validate(epoch)

            # LR step
            if self.scheduler:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']

            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('train/lr', lr, epoch)
            # Accuracy metrics
            self.writer.add_scalar('train/depth_mae', train_metrics['depth_mae'], epoch)
            self.writer.add_scalar('val/depth_mae', val_metrics['depth_mae'], epoch)
            self.writer.add_scalar('train/depth_rmse', train_metrics['depth_rmse'], epoch)
            self.writer.add_scalar('val/depth_rmse', val_metrics['depth_rmse'], epoch)
            self.writer.add_scalar('train/pos_mae', train_metrics['pos_mae'], epoch)
            self.writer.add_scalar('val/pos_mae', val_metrics['pos_mae'], epoch)
            self.writer.add_scalar('train/rel_depth_err', train_metrics['rel_depth_err'], epoch)
            self.writer.add_scalar('val/rel_depth_err', val_metrics['rel_depth_err'], epoch)
            self.writer.add_scalar('train/spike_rate', train_components.get('spike', 0), epoch)
            self.writer.add_scalar('val/spike_rate', val_components.get('spike', 0), epoch)

            self.history['train'].append({
                'epoch': epoch,
                'loss': train_loss,
                **train_components,
                **train_metrics
            })
            self.history['val'].append({
                'epoch': epoch,
                'loss': val_loss,
                **val_components,
                **val_metrics
            })

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best)

            # Summary
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time

            phase = "HEAD" if epoch < head_only_epochs else "FULL"
            spike_pct = (1.0 - train_components.get('spike', 0)) * 100
            print(f"\n{'='*80}")
            print(f"Epoch {epoch:3d} [{phase:4s}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            print(f"Depth L1: {val_components['depth']:.4f} | "
                  f"Conf BCE: {val_components['confidence']:.4f} | "
                  f"Pos MSE: {val_components['position']:.4f}")
            print(f"Val Depth MAE: {val_metrics['depth_mae']:.4f} | "
                  f"Val Depth RMSE: {val_metrics['depth_rmse']:.4f} | "
                  f"Val Rel Err: {val_metrics['rel_depth_err']:.1%}")
            print(f"Val Pos MAE: {val_metrics['pos_mae']:.4f}")
            print(f"Spike Rate: {train_components.get('spike', 0):.4f} | "
                  f"Neurons Silent: {spike_pct:.1f}%")
            print(f"Time: {epoch_time:.1f}s | "
                  f"Total: {elapsed/3600:.2f}h | "
                  f"Best Val Loss: {self.best_val_loss:.4f}")
            print(f"{'='*80}\n")

        # Save history
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        self.writer.close()

        print(f"\n🏁 Training complete!")
        print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train Drone SNN on MVSEC')

    # Data
    parser.add_argument('--data-dir', type=str, default='./data/mvsec')
    parser.add_argument('--sequences', type=str, nargs='+', default=['indoor_flying_3'])
    parser.add_argument('--batch-size', type=int, default=8)

    # Model
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--num-scales', type=int, default=3)
    parser.add_argument('--num-encoder-blocks', type=int, default=6)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--head-only-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # Pre-trained
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to N-MNIST checkpoint for fine-tuning')

    # Resume
    parser.add_argument('--resume', type=str, default=None)

    # Logging
    parser.add_argument('--log-dir', type=str, default='results/logs_drone')
    parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints_drone')

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "DRONE SNN TRAINING — MVSEC" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Device: {device.upper()}\n")

    # Create data module
    data_module = MVSECDataModule(
        data_dir=args.data_dir,
        sequences=args.sequences,
        batch_size=args.batch_size,
        num_workers=0
    )

    try:
        data_module.setup()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📋 Download MVSEC data:")
        print("   1. Go to: https://daniilidis-group.github.io/mvsec/")
        print("   2. Download Indoor Flying 1 Data (1.2 GB) + Ground Truth (2.6 GB)")
        print("   3. Place in: ./data/mvsec/indoor_flying_1/")
        print("   Google Drive: https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV")
        return

    # Create model
    model = DroneSNN(
        in_channels=2,
        hidden_channels=args.hidden_channels,
        num_scales=args.num_scales,
        num_encoder_blocks=args.num_encoder_blocks
    )

    # Create trainer
    trainer = DroneSNNTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_path=args.pretrained
    )

    # Train
    trainer.train(
        num_epochs=args.epochs,
        head_only_epochs=args.head_only_epochs,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
