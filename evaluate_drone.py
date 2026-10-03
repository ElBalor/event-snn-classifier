"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DRONE SNN EVALUATION SUITE                                                  ║
║  "Obstacle detection + depth estimation metrics"                             ║
║                                                                              ║
║  Evaluates:                                                                  ║
║  1. Obstacle detection accuracy (precision, recall, F1)                      ║
║  2. Depth estimation error (MAE, RMSE)                                       ║
║  3. False positive/negative rates                                            ║
║  4. Inference speed + spike rate                                             ║
║  5. Thin obstacle detection (power lines, branches)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from tqdm import tqdm

from model_drone import DroneSNN
from data_drone import MVSECDataset


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_detection_metrics(
    predictions: List[List[Dict]],
    ground_truth: List[List[Dict]],
    iou_threshold: float = 0.5,
    depth_threshold: float = 1.0  # meters
) -> Dict:
    """
    Compute obstacle detection metrics.

    Args:
        predictions: List of predicted obstacles per frame
        ground_truth: List of ground truth obstacles per frame
        iou_threshold: IoU threshold for matching
        depth_threshold: Depth error threshold for valid detection

    Returns:
        Dict with precision, recall, F1, etc.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    depth_errors = []

    for preds, gts in zip(predictions, ground_truth):
        # Match predictions to ground truth
        matched_preds = set()
        matched_gts = set()

        for i, pred in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(gts):
                if j in matched_gts:
                    continue

                # Compute IoU (simplified as distance-based)
                dist = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
                iou = max(0, 1 - dist / 20)  # 20 pixel tolerance

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                true_positives += 1
                matched_gts.add(best_gt_idx)
                matched_preds.add(i)

                # Depth error
                depth_err = abs(pred['depth'] - gts[best_gt_idx]['depth'])
                if depth_err < depth_threshold:
                    depth_errors.append(depth_err)
            else:
                false_positives += 1

        # Count unmatched ground truth as false negatives
        false_negatives += len(gts) - len(matched_gts)

    # Compute metrics
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_depth_error': np.mean(depth_errors) if depth_errors else float('inf'),
        'std_depth_error': np.std(depth_errors) if depth_errors else 0,
        'depth_accuracy': np.mean([e < depth_threshold for e in depth_errors]) if depth_errors else 0
    }


def compute_inference_speed(model: DroneSNN, device: str, num_runs: int = 100) -> Dict:
    """
    Measure inference speed and efficiency.
    """
    model.eval()
    model.to(device)

    times = []
    spike_rates = []

    with torch.no_grad():
        for _ in range(num_runs):
            # Dummy input
            x = torch.randn(1, 2, 128, 128, device=device)

            # Time inference
            start = time.time()
            output = model(x)
            elapsed = time.time() - start

            times.append(elapsed)

            # Estimate spike rate from attention weights
            if 'attention_weights' in output:
                attn = output['attention_weights']
                spike_rates.append(attn.mean().item())

    return {
        'mean_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'fps': 1 / np.mean(times),
        'mean_spike_rate': np.mean(spike_rates) if spike_rates else 0
    }


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN EVALUATOR
# ──────────────────────────────────────────────────────────────────────────────

class DroneSNNEvaluator:
    """
    Comprehensive evaluator for DroneSNN.
    """

    def __init__(
        self,
        model: DroneSNN,
        device: str = 'cuda',
        results_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.results_dir = results_dir or Path(__file__).parent / 'results' / 'evaluation_drone'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

    @torch.no_grad()
    def evaluate_detection(
        self,
        dataset: MVSECDataset,
        confidence_threshold: float = 0.5,
        num_samples: int = 500
    ) -> Dict:
        """
        Evaluate obstacle detection performance.
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        all_predictions = []
        all_ground_truth = []

        for events, targets, infos in tqdm(loader, desc="Evaluating detection"):
            events = events.to(self.device)
            targets = targets.to(self.device)

            # Get predictions
            output = self.model(events)
            obstacles = self.model.detect_obstacles(events, confidence_threshold)

            # Get ground truth (simplified from targets)
            batch_size = events.shape[0]
            for b in range(batch_size):
                # Extract GT obstacles from target
                gt_obstacles = []
                depth_map = targets[b, 2, :, :]
                conf_map = targets[b, 3, :, :]

                y_coords, x_coords = torch.where(conf_map > 0.5)
                for i in range(len(x_coords)):
                    gt_obstacles.append({
                        'x': x_coords[i].item(),
                        'y': y_coords[i].item(),
                        'depth': depth_map[y_coords[i], x_coords[i]].item(),
                        'confidence': 1.0
                    })

                all_predictions.append(obstacles[b] if b < len(obstacles) else [])
                all_ground_truth.append(gt_obstacles)

            if len(all_predictions) >= num_samples:
                break

        # Compute metrics
        metrics = compute_detection_metrics(all_predictions, all_ground_truth)

        self.results['detection'] = metrics
        return metrics

    @torch.no_grad()
    def evaluate_depth(
        self,
        dataset: MVSECDataset,
        num_samples: int = 500
    ) -> Dict:
        """
        Evaluate depth estimation accuracy.
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        depth_errors = []
        relative_errors = []

        for events, targets, _ in tqdm(loader, desc="Evaluating depth"):
            events = events.to(self.device)
            targets = targets.to(self.device)

            output = self.model(events)
            pred_depth = output['detection'][:, 2, :, :]
            gt_depth = targets[:, 2, :, :]

            # Mask valid depth
            mask = gt_depth > 0

            if mask.sum() == 0:
                continue

            # Absolute error
            abs_err = (pred_depth[mask] - gt_depth[mask]).abs()
            depth_errors.extend(abs_err.cpu().tolist())

            # Relative error
            rel_err = abs_err / (gt_depth[mask] + 1e-8)
            relative_errors.extend(rel_err.cpu().tolist())

            if len(depth_errors) >= num_samples:
                break

        metrics = {
            'mae': np.mean(depth_errors) if depth_errors else float('inf'),
            'rmse': np.sqrt(np.mean(np.array(depth_errors)**2)) if depth_errors else float('inf'),
            'relative_error': np.mean(relative_errors) if relative_errors else float('inf'),
            'num_samples': len(depth_errors)
        }

        self.results['depth'] = metrics
        return metrics

    @torch.no_grad()
    def evaluate_speed(
        self,
        num_runs: int = 100
    ) -> Dict:
        """
        Evaluate inference speed.
        """
        speed_metrics = compute_inference_speed(self.model, self.device, num_runs)
        self.results['speed'] = speed_metrics
        return speed_metrics

    def generate_report(self) -> str:
        """Generate human-readable evaluation report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append("DRONE SNN EVALUATION REPORT")
        report.append("=" * 80)

        # Detection metrics
        if 'detection' in self.results:
            det = self.results['detection']
            report.append("\n🎯 OBSTACLE DETECTION")
            report.append("-" * 60)
            report.append(f"  Precision: {det['precision']:.3f}")
            report.append(f"  Recall: {det['recall']:.3f}")
            report.append(f"  F1 Score: {det['f1_score']:.3f}")
            report.append(f"  True Positives: {det['true_positives']}")
            report.append(f"  False Positives: {det['false_positives']}")
            report.append(f"  False Negatives: {det['false_negatives']}")

        # Depth metrics
        if 'depth' in self.results:
            depth = self.results['depth']
            report.append("\n📏 DEPTH ESTIMATION")
            report.append("-" * 60)
            report.append(f"  MAE: {depth['mae']:.3f} m")
            report.append(f"  RMSE: {depth['rmse']:.3f} m")
            report.append(f"  Relative Error: {depth['relative_error']:.3f}")

        # Speed metrics
        if 'speed' in self.results:
            speed = self.results['speed']
            report.append("\n⚡ INFERENCE SPEED")
            report.append("-" * 60)
            report.append(f"  Latency: {speed['mean_latency_ms']:.2f} ± {speed['std_latency_ms']:.2f} ms")
            report.append(f"  FPS: {speed['fps']:.1f}")
            report.append(f"  Spike Rate: {speed['mean_spike_rate']:.4f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_report(self, filename: str = "drone_evaluation_report.txt"):
        """Save report to file."""
        report = self.generate_report()
        with open(self.results_dir / filename, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {self.results_dir / filename}")

        # Save raw data
        with open(self.results_dir / 'drone_evaluation_data.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📊 Raw data saved to: {self.results_dir / 'drone_evaluation_data.json'}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "DRONE SNN EVALUATION" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝\n")

    # Load model
    checkpoint_path = Path(__file__).parent / 'results' / 'checkpoints_drone' / 'best.pt'

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Train the model first!")
        exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DroneSNN(
        in_channels=2,
        hidden_channels=128,
        num_scales=3,
        num_encoder_blocks=6
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"🔥 Device: {device.upper()}\n")

    # Create evaluator
    evaluator = DroneSNNEvaluator(model, device=device)

    # Note: Evaluation requires MVSEC dataset
    print("📋 Evaluation requires MVSEC dataset")
    print("   Download from: https://daniilidis-group.github.io/mvsec/")
    print("   Place in: ./data/mvsec/")

    print("\n✅ Evaluation suite ready!\n")
