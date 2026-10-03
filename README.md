# 🧠 Experiment 2: Event-Based Vision SNN Classifier

## *"Neuromorphic classification with surrogate gradients"*

---

## 🔮 Overview

This is a **Spiking Neural Network (SNN)** for event-based vision classification. Unlike standard neural networks, SNNs process **binary spikes** over time, making them:

- **Energy efficient** - Only active when events occur
- **Neuromorphic** - Runs on brain-inspired hardware (Loihi, SpiNNaker)
- **Temporal** - Processes time-encoded information naturally

**Key innovations:**

1. **Surrogate Gradient Learning** - Train through non-differentiable spikes
2. **Energy Efficiency** - "Whisper the answer" - minimal spiking
3. **Event-Based Vision** - N-MNIST dataset (DVS camera events)

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    N-MNIST EVENTS                               │
│  70,000 event-based digit recordings                            │
│  34 × 34 pixels, 2 polarities, ~300ms duration                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 VOXEL GRID ENCODING                             │
│  Events → [2, 34, 34, T] tensor                                 │
│  Time-binned representation (T=10 bins)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           SPIKING CONVOLUTIONAL ENCODER                         │
│  Conv1: 2→16 channels, LIF neurons                              │
│  Conv2: 16→32 channels, stride=2, LIF                           │
│  Conv3: 32→64 channels, stride=2, LIF                           │
│  Output: [B, 64, 9, 9] spike trains                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               SPIKE CLASSIFIER                                  │
│  Spike count encoding → FC(4096→256) → FC(256→128) → FC(128→10) │
│  Output: Digit class (0-9)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Surrogate Gradient Learning

### The Problem

Spikes are **binary** (0 or 1) → **non-differentiable** → can't backpropagate.

```python
spike = 1 if membrane > threshold else 0  # Not differentiable!
```

### Our Solution

**Surrogate gradient:** Use smooth sigmoid approximation during backward pass.

```python
# Forward: binary spike
spike = (membrane > threshold).float()

# Backward: smooth gradient
grad = sigmoid(steepness * (membrane - threshold))
```

This allows gradients to flow through spikes while keeping binary forward pass.

---

## ⚡ Energy Efficiency ("Whisper the Answer")

### The Goal

**High accuracy with minimal spiking.** Each spike costs energy - we want to "whisper" the answer, not shout it.

### Implementation

```python
# Loss = CrossEntropy + λ × SpikeRate
loss = CE_loss + 0.001 × avg_spike_rate
```

This penalizes excessive firing, encouraging sparse coding.

### Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| **Accuracy** | >90% | Classification performance |
| **Spike Rate** | <0.1 | Average neuron firing rate |
| **Spike Efficiency** | >100 | Correct classifications per spike |

---

## 📁 Project Structure

```
02_event_snn_classifier/
├── model.py              # SNN with surrogate gradients
├── data.py               # N-MNIST loader (tonic)
├── train.py              # Training loop
├── evaluate.py           # Evaluation metrics
├── requirements.txt      # Dependencies
└── results/
    ├── checkpoints/      # Saved models
    ├── logs/             # TensorBoard logs
    └── evaluation/       # Results & plots
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install snntorch tonic
```

### 2. Download N-MNIST (Auto-downloads on first run)

```bash
# First run will download ~50MB dataset
python train.py --epochs 50
```

### 3. Train

```bash
# Standard training
python train.py --epochs 100 --batch-size 64 --lr 1e-3

# With custom energy penalty
python train.py --epochs 100 --spike-loss-weight 0.01

# Resume from checkpoint
python train.py --epochs 100 --resume results/checkpoints/latest.pt
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint results/checkpoints/best.pt
```

### 5. Monitor

```bash
tensorboard --logdir results/logs
```

---

## 📊 Expected Results

### Performance (RTX 3080)

| Metric | Value |
|--------|-------|
| Training Speed | ~50 samples/sec |
| Memory | ~2 GB |
| Epoch Time | ~2 min |

### Accuracy

| Epoch | Accuracy | Spike Rate |
|-------|----------|------------|
| 10 | 60-70% | 0.15 |
| 50 | 85-90% | 0.08 |
| 100 | 90-93% | 0.05 |

### Energy Efficiency

| Spike Loss Weight | Accuracy | Avg Spikes/Sample |
|-------------------|----------|-------------------|
| 0.0 (no penalty) | 93% | 2000 |
| 0.001 (default) | 92% | 1200 |
| 0.01 (high) | 88% | 500 |

---

## 🎮 Advanced Usage

### Hyperparameter Tuning

```bash
# Adjust LIF decay (memory)
python train.py --lif-decay 0.9  # Less memory
python train.py --lif-decay 0.99 # More memory

# Adjust threshold (sparsity)
python train.py --lif-threshold 0.5  # More spikes
python train.py --lif-threshold 1.5  # Fewer spikes

# Adjust energy penalty
python train.py --spike-loss-weight 0.01  # Strong penalty
```

### Custom Time Bins

```bash
# More time bins = better temporal resolution
python train.py --time-bins 20
```

---

## 📚 Key Concepts

### LIF Neuron (Leaky Integrate-and-Fire)

```
τ_m × dV/dt = -V + I_input    # Membrane integration
if V > threshold: spike = 1   # Fire
V = V_reset                   # Reset
```

### Surrogate Gradient

```python
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > threshold).float()  # Binary
    
    @staticmethod
    def backward(ctx, grad):
        return grad * sigmoid(x)  # Smooth
```

### Spike Rate Regularization

```python
loss = CrossEntropy(logits, labels) + λ × mean(spikes)
```

---

## 🏆 Research Applications

This SNN approach enables:

1. **Neuromorphic Computing** - Runs on Loihi, SpiNNaker, DVS cameras
2. **Event-Based Vision** - Low-latency, high dynamic range
3. **Edge AI** - Ultra-low power consumption
4. **Temporal Processing** - Natural handling of time-encoded data

---

## 📚 References

### Key Papers

1. **Surrogate Gradients:**
   - Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks" (2019)

2. **SNNs for Vision:**
   - Rueckauer et al., "Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks" (2017)

3. **N-MNIST Dataset:**
   - Orchard et al., "N-MNIST: A Neuromorphic Vision Dataset" (2015)

4. **Energy-Efficient SNNs:**
   - Diehl & Cook, "Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity" (2015)

---

## 🔧 Troubleshooting

### tonic Import Error

```bash
pip install tonic
```

### CUDA Out of Memory

```bash
python train.py --batch-size 32
```

### Slow Training

- Use GPU (CUDA) - CPU is ~10x slower
- Reduce time bins: `--time-bins 5`
- Reduce batch size

### Low Accuracy

- Train longer (100+ epochs)
- Adjust learning rate: `--lr 5e-4`
- Check surrogate gradient is working (gradients should flow)

---

## 🚁 DRONE SNN — OBSTACLE DETECTION (3M Parameters)

### Overview

A **3M parameter Spiking Neural Network** for real-time obstacle detection on autonomous drones using event cameras.

**Fine-tuned from N-MNIST model (97.11% accuracy) → Drone obstacle detection**

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT CAMERA INPUT                           │
│  128 × 128 pixels, 2 polarities, 10 time bins                   │
│  (Prophesee/iniVation DVS sensor)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEM (128×128 → 32×32)                                         │
│  Conv2d(2, 64, stride=2) → Conv2d(64, 128, stride=2)           │
│  Parameters: ~150K                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  MULTI-SCALE TCN ENCODER (3 branches × 6 blocks, 128 channels) │
│  Scale 1: dilation 1,2,4,8,16,32                                │
│  Scale 2: dilation 2,4,8,16,32,64                               │
│  Scale 3: dilation 4,8,16,32,64,128                             │
│  Parameters: ~1.8M                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DETECTION HEAD                                                 │
│  Conv2d(128→256→128→64→4)                                       │
│  Output: x, y, depth, confidence per spatial cell               │
│  Parameters: ~450K                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: (N, 4) per detected obstacle                           │
│  - x, y: Position (0-128 pixels)                                │
│  - depth: Distance (meters)                                     │
│  - confidence: 0-1                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Specs

| Metric | Value |
|--------|-------|
| **Total Parameters** | ~3.0M |
| **Input Resolution** | 128×128 (full DVS) |
| **Spike Rate** | ~0.04 (96% neurons silent) |
| **Power on Loihi** | ~80mW |
| **Inference Speed** | ~10-20ms per frame |
| **Memory** | ~35 MB |

### Quick Start

```bash
# 1. Download MVSEC dataset
# Go to: https://daniilidis-group.github.io/mvsec/
# Download outdoor_day1 + outdoor_day2
# Place in: ./data/mvsec/

# 2. Fine-tune from N-MNIST checkpoint
python train_drone.py \
    --epochs 100 \
    --batch-size 8 \
    --pretrained results/checkpoints/best.pt \
    --sequences outdoor_day1 outdoor_day2

# 3. Evaluate
python evaluate_drone.py --checkpoint results/checkpoints_drone/best.pt
```

### Training Strategy

| Phase | Epochs | What Happens |
|-------|--------|--------------|
| **Phase 1: Head-only** | 0-10 | Encoder frozen, detection head learns |
| **Phase 2: Full fine-tuning** | 10-100 | All layers unfrozen, end-to-end training |

### Expected Results

| Metric | Target |
|--------|--------|
| **Detection F1** | >0.85 |
| **Depth MAE** | <0.5m |
| **False Positives** | <5% |
| **Inference Latency** | <20ms |

---

## 📄 License

MIT License - Use freely for research and commercial projects.

---

<div align="center">

**"In the spike, we trust"**

*Eric Yaka || The Digital Necromancer*

*Part of the Temporal Signal Filter Lab*

</div>
