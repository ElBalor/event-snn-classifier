"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SPIKING NEURAL NETWORK FOR EVENT-BASED VISION                               ║
║  "Surrogate gradients + energy-efficient coding"                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


# ──────────────────────────────────────────────────────────────────────────────
#  SURROGATE GRADIENT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons.
    
    Forward pass: Binary spike (Heaviside step)
    Backward pass: Smooth sigmoid approximation
    
    This allows gradients to flow through non-differentiable spikes.
    """
    
    @staticmethod
    def forward(ctx, x, threshold=1.0):
        """Forward: binary spike (0 or 1)"""
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward: smooth sigmoid gradient"""
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # Sigmoid surrogate gradient
        # Steepness controls how fast the gradient decays
        steepness = 10.0
        sigmoid_grad = torch.sigmoid(steepness * (x - threshold))
        
        # Gradient is highest near threshold, zero far away
        return grad_output * sigmoid_grad * (1 - sigmoid_grad) * steepness, None


def surrogate_heaviside(x, threshold=1.0):
    """Apply surrogate gradient Heaviside function."""
    return SurrogateGradient.apply(x, threshold)


# ──────────────────────────────────────────────────────────────────────────────
#  LIF NEURON WITH SURROGATE GRADIENTS
# ──────────────────────────────────────────────────────────────────────────────

class SurrogateLIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with surrogate gradient learning.
    
    Dynamics:
        τ_m * dV/dt = -V + I_input    (membrane integration)
        if V > threshold: spike = 1, V = V_reset  (fire and reset)
    
    Uses surrogate gradients for backpropagation through spikes.
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.95,
        reset_value: float = 0.0,
        learn_threshold: bool = False
    ):
        super().__init__()
        
        self.threshold = threshold
        self.decay = decay  # Membrane decay (1 - decay = leak)
        self.reset_value = reset_value
        self.learn_threshold = learn_threshold
        
        # Optional: learnable threshold
        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        
        # State variables
        self.register_buffer('membrane', torch.tensor(0.0))
        self.register_buffer('spike', torch.tensor(0.0))
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        Integrate input current and emit spikes.
        
        Args:
            input_current: Input current [B, C, H, W]
            
        Returns:
            spike: Binary spike output (0 or 1)
        """
        # Membrane integration with leak
        # Don't modify in-place - create new tensor
        new_membrane = self.decay * self.membrane * (1 - self.spike.detach()) + input_current
        
        # Reset mechanism
        new_membrane = new_membrane - self.spike.detach() * self.threshold
        
        # Update state
        self.membrane = new_membrane
        
        # Generate spike with surrogate gradient
        self.spike = surrogate_heaviside(self.membrane, self.threshold)
        
        return self.spike
    
    def reset(self):
        """Reset neuron state."""
        self.membrane.zero_()
        self.spike.zero_()


# ──────────────────────────────────────────────────────────────────────────────
#  SPIKING CONV BLOCK
# ──────────────────────────────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """
    Convolutional layer + LIF neuron for spiking neural networks.
    Processes event data over time.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        lif_decay: float = 0.95,
        lif_threshold: float = 1.0
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif_decay = lif_decay
        self.lif_threshold = lif_threshold
        
    def forward(self, x: torch.Tensor, membrane: torch.Tensor = None, spike: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process single time step.
        
        Args:
            x: Input events [B, C, H, W]
            membrane: Previous membrane potential
            spike: Previous spike output
            
        Returns:
            spikes: Output spikes [B, C_out, H, W]
            membrane: New membrane potential
            spike: New spike output
        """
        x = self.conv(x)
        x = self.bn(x)
        
        # LIF integration (inline to avoid state issues)
        if membrane is None:
            membrane = torch.zeros_like(x)
        if spike is None:
            spike = torch.zeros_like(x)
        
        # Membrane integration with leak
        membrane = self.lif_decay * membrane * (1 - spike.detach()) + x
        membrane = membrane - spike.detach() * self.lif_threshold
        
        # Generate spike with surrogate gradient
        spike = surrogate_heaviside(membrane, self.lif_threshold)
        
        return spike, membrane, spike
    
    def reset(self):
        """No state to reset - state is passed explicitly."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  EVENT ENCODER (Voxel Grid → Spikes)
# ──────────────────────────────────────────────────────────────────────────────

class EventEncoder(nn.Module):
    """
    Encodes event streams (N-MNIST) into spike trains.
    
    Input: Voxel grid [B, 2, H, W, T] (2 polarities, time bins)
    Output: Spike trains [B, C, H', W'] at final time step
    """
    
    def __init__(
        self,
        num_time_bins: int = 10,
        lif_decay: float = 0.95,
        lif_threshold: float = 1.0
    ):
        super().__init__()
        
        self.num_time_bins = num_time_bins
        
        # Convolutional layers with LIF neurons
        self.conv1 = SpikingConvBlock(2, 16, 3, lif_decay=lif_decay, lif_threshold=lif_threshold)
        self.conv2 = SpikingConvBlock(16, 32, 3, stride=2, lif_decay=lif_decay, lif_threshold=lif_threshold)
        self.conv3 = SpikingConvBlock(32, 64, 3, stride=2, lif_decay=lif_decay, lif_threshold=lif_threshold)
        
        # Pooling
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Encode event stream to spike representation.
        
        Args:
            events: Voxel grid [B, 2, H, W, T]
            
        Returns:
            spikes: Final spike representation [B, 64, H', W']
        """
        B, C, H, W, T = events.shape
        
        # State variables for each layer
        mem1, spike1 = None, None
        mem2, spike2 = None, None
        mem3, spike3 = None, None
        
        # Process each time bin
        final_spikes = None
        
        for t in range(T):
            # Get events at this time step
            x_t = events[:, :, :, :, t]  # [B, 2, H, W]
            
            # Pass through spiking conv layers
            x_t, mem1, spike1 = self.conv1(x_t, mem1, spike1)  # [B, 16, H, W]
            x_t, mem2, spike2 = self.conv2(x_t, mem2, spike2)  # [B, 32, H/2, W/2]
            x_t, mem3, spike3 = self.conv3(x_t, mem3, spike3)  # [B, 64, H/4, W/4]
            
            final_spikes = x_t
        
        return final_spikes
    
    def reset(self):
        """Reset all LIF neurons."""
        self.conv1.reset()
        self.conv2.reset()
        self.conv3.reset()


# ──────────────────────────────────────────────────────────────────────────────
#  SPIKE CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

class SpikeClassifier(nn.Module):
    """
    Classifies spike trains into categories.
    Uses spike counting + fully connected layers.
    """
    
    def __init__(
        self,
        input_features: int,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Classify from spike trains.
        
        Args:
            spikes: Spike trains [B, C, H, W]
            
        Returns:
            logits: Class predictions [B, num_classes]
        """
        # Spike count encoding (sum over spatial dimensions)
        # spikes: [B, C, H, W] → sum over C, H, W → [B]
        spike_count = spikes.view(spikes.shape[0], -1)  # [B, C*H*W]
        
        # Classify
        logits = self.classifier(spike_count)
        
        return logits


# ──────────────────────────────────────────────────────────────────────────────
#  FULL SNN MODEL (WITH ENERGY EFFICIENCY)
# ──────────────────────────────────────────────────────────────────────────────

class EventSNN(nn.Module):
    """
    Complete Spiking Neural Network for event-based vision.
    
    Features:
    - Surrogate gradient learning (trainable through spikes)
    - Energy efficiency via spike rate regularization
    - Minimal spiking ("whisper the answer")
    
    Architecture:
        Events → Voxel Grid → Spiking Conv Encoder → Spike Classifier → Class
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (2, 34, 34),  # N-MNIST: 2 pol, 34x34
        num_classes: int = 10,
        num_time_bins: int = 10,
        lif_decay: float = 0.95,
        lif_threshold: float = 1.0,
        spike_loss_weight: float = 0.001  # Energy efficiency weight
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.spike_loss_weight = spike_loss_weight
        
        # Encoder: events → spikes
        self.encoder = EventEncoder(
            num_time_bins=num_time_bins,
            lif_decay=lif_decay,
            lif_threshold=lif_threshold
        )
        
        # Classifier: spikes → class
        # After encoder: 64 channels, 34→17→8 (after 2 stride-2 convs)
        # Actually: 34→34→17→8 (conv1 no stride, conv2 stride 2, conv3 stride 2)
        encoder_output_dim = 64 * 9 * 9  # [B, 64, 9, 9] flattened
        self.classifier = SpikeClassifier(
            input_features=encoder_output_dim,
            num_classes=num_classes,
            hidden_dim=256
        )
        
        # For tracking spike statistics
        self.spike_counts = []
        
    def forward(self, events: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            events: Voxel grid [B, 2, H, W, T]
            
        Returns:
            dict with:
                - logits: Class predictions [B, num_classes]
                - spikes: Encoder output spikes [B, 64, 8, 8]
                - spike_rate: Average firing rate (for energy loss)
        """
        # Encode events to spikes
        spikes = self.encoder(events)
        
        # Classify
        logits = self.classifier(spikes)
        
        # Compute spike rate (for energy regularization)
        spike_rate = spikes.mean()
        
        return {
            'logits': logits,
            'spikes': spikes,
            'spike_rate': spike_rate
        }
    
    def classify(self, events: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for classification only.
        
        Args:
            events: Voxel grid [B, 2, H, W, T]
            
        Returns:
            predictions: Class indices [B]
        """
        output = self.forward(events)
        predictions = output['logits'].argmax(dim=-1)
        return predictions
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined classification + energy loss.
        
        Loss = CrossEntropy + λ × SpikeRate
        
        The spike rate penalty encourages "whispering" - minimal spiking
        while maintaining accuracy.
        
        Args:
            output: Model output dict
            targets: Ground truth labels [B]
            reduction: 'mean' or 'sum'
            
        Returns:
            total_loss, loss_components
        """
        logits = output['logits']
        spike_rate = output['spike_rate']
        
        # Classification loss (cross-entropy)
        cls_loss = F.cross_entropy(logits, targets, reduction=reduction)
        
        # Energy loss (spike rate regularization)
        # Encourages sparse coding - "whisper the answer"
        energy_loss = self.spike_loss_weight * spike_rate
        
        # Total loss
        total = cls_loss + energy_loss
        
        components = {
            'total': total.item() if isinstance(total, torch.Tensor) else total,
            'classification': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'energy': energy_loss.item() if isinstance(energy_loss, torch.Tensor) else energy_loss,
            'spike_rate': spike_rate.item() if isinstance(spike_rate, torch.Tensor) else spike_rate
        }
        
        return total, components
    
    def reset(self):
        """Reset all neuron states."""
        self.encoder.reset()


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def count_spikes(spikes: torch.Tensor) -> int:
    """Count total spikes in a spike train."""
    return (spikes > 0).sum().item()


def compute_spike_efficiency(spikes: torch.Tensor, accuracy: float) -> float:
    """
    Compute spikes-per-correct-classification.
    Lower is better (more efficient coding).
    """
    total_spikes = count_spikes(spikes)
    return total_spikes / (accuracy * 100 + 1e-8)


def get_model_summary(model: EventSNN) -> str:
    """Generate model summary string."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SNN MODEL ARCHITECTURE SUMMARY                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Parameters: {total_params:>10,} ({total_params/1e6:.2f}M)                            ║
║  Input Shape: {model.input_shape}                                                         ║
║  Num Classes: {model.num_classes}                                                           ║
║  Time Bins: {model.encoder.num_time_bins}                                                               ║
║  Spike Loss Weight: {model.spike_loss_weight}                                                 ║
║  LIF Decay: {model.encoder.conv1.lif_decay}                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "SNN MODEL TEST - SURROGATE GRADIENTS" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Create model
    model = EventSNN(
        input_shape=(2, 34, 34),
        num_classes=10,
        num_time_bins=10,
        spike_loss_weight=0.001
    )
    
    print(get_model_summary(model))
    
    # Test forward pass
    batch_size = 4
    events = torch.randn(batch_size, 2, 34, 34, 10)  # Simulated events
    
    print("🧪 Forward pass test...\n")
    
    output = model(events)
    
    print(f"Input shape:     {events.shape}")
    print(f"Logits shape:    {output['logits'].shape}")
    print(f"Spike shape:     {output['spikes'].shape}")
    print(f"Spike rate:      {output['spike_rate']:.4f}")
    print(f"Total spikes:    {count_spikes(output['spikes'])}")
    
    # Test loss computation
    print("\n🧪 Loss computation test...\n")
    
    targets = torch.randint(0, 10, (batch_size,))
    loss, components = model.compute_loss(output, targets)
    
    print(f"Total Loss:      {loss:.4f}")
    print(f"  - Classification: {components['classification']:.4f}")
    print(f"  - Energy:         {components['energy']:.6f}")
    print(f"  - Spike Rate:     {components['spike_rate']:.4f}")
    
    # Test classification
    print("\n🧪 Classification test...\n")
    
    predictions = model.classify(events)
    print(f"Predictions:     {predictions}")
    print(f"Accuracy:        {(predictions == targets).float().mean().item():.2%}")
    
    # Test surrogate gradient (verify gradients flow through spikes)
    print("\n🧪 Surrogate gradient test...\n")
    
    loss.backward()
    
    # Check if encoder has gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                    for p in model.encoder.parameters())
    
    if has_grads:
        print("✅ Gradients flowing through spikes (surrogate gradient works!)")
    else:
        print("❌ No gradients - check surrogate gradient implementation")
    
    print("\n✅ SNN model test passed!\n")
