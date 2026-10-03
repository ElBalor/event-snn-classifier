"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SNN EVALUATION                                                              ║
║  "Accuracy + spike efficiency metrics"                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List

from model import EventSNN, count_spikes, compute_spike_efficiency
from data import NMNISTDataModule


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class SNNEvaluator:
    """
    Evaluates trained SNN on N-MNIST.
    """
    
    def __init__(
        self,
        model: EventSNN,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cpu',
        results_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.test_loader = test_loader
        self.results_dir = results_dir or Path(__file__).parent / 'results' / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run full evaluation."""
        print("\n🔬 Running evaluation...\n")
        
        all_predictions = []
        all_labels = []
        all_spike_counts = []
        all_spike_rates = []
        
        total_correct = 0
        total_samples = 0
        
        for events, labels, infos in self.test_loader:
            events = events.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            output = self.model(events)
            logits = output['logits']
            spike_rate = output['spike_rate']
            spikes = output['spikes']
            
            # Get predictions
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_spike_counts.append(count_spikes(spikes))
            all_spike_rates.append(spike_rate.item())
            
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
        
        # Compute metrics
        accuracy = 100.0 * total_correct / total_samples
        avg_spike_count = np.mean(all_spike_counts)
        avg_spike_rate = np.mean(all_spike_rates)
        spike_efficiency = compute_spike_efficiency(
            torch.zeros(1, 64, 9, 9).sum() + sum(all_spike_counts),
            accuracy / 100
        )
        
        # Per-class accuracy
        per_class_acc = {}
        for cls in range(10):
            cls_mask = [l == cls for l in all_labels]
            cls_correct = sum(1 for p, l, m in zip(all_predictions, all_labels, cls_mask) if p == l and m)
            cls_total = sum(cls_mask)
            per_class_acc[cls] = cls_correct / cls_total if cls_total > 0 else 0
        
        # Confusion matrix
        confusion = self._compute_confusion_matrix(all_predictions, all_labels)
        
        results = {
            'accuracy': accuracy,
            'avg_spike_count': avg_spike_count,
            'avg_spike_rate': avg_spike_rate,
            'spike_efficiency': spike_efficiency,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion.tolist(),
            'total_samples': total_samples
        }
        
        return results
    
    def _compute_confusion_matrix(self, predictions: List[int], labels: List[int]) -> np.ndarray:
        """Compute confusion matrix."""
        num_classes = 10
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for pred, label in zip(predictions, labels):
            confusion[label, pred] += 1
        
        return confusion
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("SNN EVALUATION REPORT")
        report.append("=" * 80)
        
        report.append(f"\n📊 OVERALL METRICS")
        report.append("-" * 60)
        report.append(f"  Accuracy:        {results['accuracy']:.2f}%")
        report.append(f"  Avg Spike Count: {results['avg_spike_count']:.1f}")
        report.append(f"  Avg Spike Rate:  {results['avg_spike_rate']:.4f}")
        report.append(f"  Spike Efficiency: {results['spike_efficiency']:.2f} correct/spike")
        
        report.append(f"\n📊 PER-CLASS ACCURACY")
        report.append("-" * 60)
        for cls, acc in results['per_class_accuracy'].items():
            bar = "█" * int(acc * 20)
            report.append(f"  Class {cls}: {acc*100:5.1f}%  {bar}")
        
        report.append(f"\n📊 CONFUSION MATRIX")
        report.append("-" * 60)
        confusion = np.array(results['confusion_matrix'])
        
        # Print header
        report.append("       " + "  ".join(f"{i:3d}" for i in range(10)))
        report.append("       " + "-" * 35)
        
        # Print rows
        for i, row in enumerate(confusion):
            report.append(f"  {i:2d} |  " + "  ".join(f"{v:3d}" for v in row))
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict):
        """Save results to files."""
        # Save JSON
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save report
        report = self.generate_report(results)
        with open(self.results_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(np.array(results['confusion_matrix']))
        
        print(f"\n💾 Results saved to: {self.results_dir}")
    
    def _plot_confusion_matrix(self, confusion: np.ndarray):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize for display
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_norm = confusion / (row_sums + 1e-8)
        
        im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Count')
        
        # Labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        # Add numbers
        for i in range(10):
            for j in range(10):
                text = ax.text(j, i, str(confusion[i, j]),
                              ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SNN')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/nmnist')
    parser.add_argument('--time-bins', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='results/evaluation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 22 + "SNN EVALUATION" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Device: {device.upper()}\n")
    
    # Load model
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = EventSNN(
        num_classes=10,
        num_time_bins=args.time_bins
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model loaded\n")
    
    # Load data
    print("📊 Loading test data...")
    data_module = NMNISTDataModule(
        root=args.data_dir,
        num_time_bins=args.time_bins,
        batch_size=args.batch_size,
        num_workers=0
    )
    data_module.setup()
    
    # Create evaluator
    evaluator = SNNEvaluator(
        model=model,
        test_loader=data_module.test_dataloader(),
        device=device,
        results_dir=Path(args.output_dir)
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    evaluator.save_results(results)
    
    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
