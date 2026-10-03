"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SNN TRAINING WITH SURROGATE GRADIENTS                                       ║
║  "Energy-efficient event classification"                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json
from tqdm import tqdm

import sys
from pathlib import Path
# Add experiments root to sys.path for telemetry import
sys.path.append(str(Path(__file__).parent.parent))
from telemetry_client import TelemetryClient

# ──────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class SNNTrainer:
    """
    Trains SNN with surrogate gradients and energy efficiency.
    """
    
    def __init__(
        self,
        model: EventSNN,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu',
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        self.device = device
        self.model = model.to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-5
        )
        
        # Loss: Cross-entropy + spike rate regularization (already in model)
        self.criterion = nn.CrossEntropyLoss()
        
        # Logging
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / 'results' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(__file__).parent / 'results' / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.history = {'train': [], 'val': []}
        self.best_val_acc = 0
        
        # Telemetry for 3D Dashboard
        self.telemetry = TelemetryClient("Event_SNN_Classifier")
        
        print(f"\n🔥 Training on: {device.upper()}")
        print(get_model_summary(model))
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_spike_rate = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [TRAIN]')
        
        for events, labels, _ in pbar:
            events = events.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(events)
            
            logits = output['logits']
            spike_rate = output['spike_rate']
            
            # Compute loss (includes energy regularization)
            loss, components = self.model.compute_loss(output, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            _, predicted = logits.max(1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            total_spike_rate += spike_rate.item()
            num_batches += 1
            
            # Send real-time telemetry
            if num_batches % 5 == 0:
                self.telemetry.log_metrics(epoch, {
                    "accuracy": correct / labels.size(0),
                    "loss": loss.item()
                })
                self.telemetry.log_spikes(spike_rate.item())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / labels.size(0):.1f}%',
                'spikes': f'{spike_rate.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * total_correct / total_samples
        avg_spike_rate = total_spike_rate / num_batches
        
        return avg_loss, accuracy, avg_spike_rate
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float, Dict]:
        """Validate on test set."""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_spike_rate = 0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [VAL]  ')
        
        for events, labels, _ in pbar:
            events = events.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            output = self.model(events)
            
            logits = output['logits']
            spike_rate = output['spike_rate']
            
            # Compute loss
            loss, components = self.model.compute_loss(output, labels)
            
            # Metrics
            _, predicted = logits.max(1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            total_spike_rate += spike_rate.item()
            num_batches += 1
            
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / labels.size(0):.1f}%'
            })
        
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * total_correct / total_samples
        avg_spike_rate = total_spike_rate / num_batches
        
        # Per-class accuracy
        per_class_acc = {}
        for cls in range(10):
            cls_mask = [l == cls for l in all_labels]
            cls_correct = sum(1 for p, l, m in zip(all_predictions, all_labels, cls_mask) if p == l and m)
            cls_total = sum(cls_mask)
            per_class_acc[cls] = cls_correct / cls_total if cls_total > 0 else 0
        
        metrics = {
            'per_class_accuracy': per_class_acc,
            'avg_spike_rate': avg_spike_rate,
            'spike_efficiency': total_correct / (sum(all_predictions) + 1e-8)  # Correct per spike
        }
        
        return avg_loss, accuracy, avg_spike_rate, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"   💾 New best model saved! (Acc: {self.best_val_acc:.2f}%)")
    
    def train(
        self,
        num_epochs: int = 100,
        resume_from: Optional[str] = None
    ):
        """Full training loop."""
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from:
            print(f"\n📥 Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            print(f"Resumed from epoch {start_epoch}, best acc: {self.best_val_acc:.2f}%")
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...\n")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_spike_rate = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_spike_rate, val_metrics = self.validate(epoch)
            
            # Learning rate step
            if self.scheduler:
                self.scheduler.step()
            
            # Log to tensorboard
            lr = self.optimizer.param_groups[0]['lr']
            
            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/accuracy', train_acc, epoch)
            self.writer.add_scalar('train/spike_rate', train_spike_rate, epoch)
            self.writer.add_scalar('train/lr', lr, epoch)
            
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/accuracy', val_acc, epoch)
            self.writer.add_scalar('val/spike_rate', val_spike_rate, epoch)
            self.writer.add_scalar('val/spike_efficiency', val_metrics['spike_efficiency'], epoch)
            
            # Save to history
            self.history['train'].append({
                'epoch': epoch,
                'loss': train_loss,
                'accuracy': train_acc,
                'spike_rate': train_spike_rate
            })
            
            self.history['val'].append({
                'epoch': epoch,
                'loss': val_loss,
                'accuracy': val_acc,
                'spike_rate': val_spike_rate,
                'spike_efficiency': val_metrics['spike_efficiency']
            })
            
            # Checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)
            
            # Save history
            self._save_history()
            
            # Progress summary
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 80)
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%")
            print(f"Train Spikes: {train_spike_rate:.4f} | "
                  f"Val Spikes: {val_spike_rate:.4f} | "
                  f"Spike Eff: {val_metrics['spike_efficiency']:.2f}")
            print(f"Time: {epoch_time:.1f}s | "
                  f"Total: {elapsed/3600:.2f}h | "
                  f"Best Val Acc: {self.best_val_acc:.2f}%")
            print("=" * 80 + "\n")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🏁 Training complete!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final history
        self._save_history()
        self.writer.close()
        
        return self.history
    
    def _save_history(self):
        """Save training history to JSON."""
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(convert(self.history), f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train SNN on N-MNIST')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data/nmnist')
    parser.add_argument('--time-bins', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    
    # Model
    parser.add_argument('--lif-decay', type=float, default=0.95)
    parser.add_argument('--lif-threshold', type=float, default=1.0)
    parser.add_argument('--spike-loss-weight', type=float, default=0.001)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None)
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='results/logs')
    parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 22 + "SNN TRAINING - N-MNIST" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Device: {device.upper()}\n")
    
    # Create data module
    data_module = NMNISTDataModule(
        root=args.data_dir,
        num_time_bins=args.time_bins,
        batch_size=args.batch_size,
        num_workers=0  # Windows compatibility
    )
    data_module.setup()
    
    # Create model
    model = EventSNN(
        num_classes=10,
        num_time_bins=args.time_bins,
        lif_decay=args.lif_decay,
        lif_threshold=args.lif_threshold,
        spike_loss_weight=args.spike_loss_weight
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    
    # Create trainer
    trainer = SNNTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume
    )
    
    print("\n🎉 Training finished!")
    print("📊 Run: tensorboard --logdir", args.log_dir)
    print("\n")


if __name__ == "__main__":
    main()
