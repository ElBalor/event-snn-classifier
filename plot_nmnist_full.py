"""Extract full N-MNIST training curves from ALL TensorBoard logs."""
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('dark_background')
mpl.rcParams['axes.facecolor'] = '#1a1a2e'
mpl.rcParams['figure.facecolor'] = '#0f0f1a'
mpl.rcParams['grid.color'] = '#333355'
mpl.rcParams['text.color'] = '#e0e0e0'
mpl.rcParams['axes.labelcolor'] = '#e0e0e0'
mpl.rcParams['xtick.color'] = '#e0e0e0'
mpl.rcParams['ytick.color'] = '#e0e0e0'

LOGS_DIR = Path(__file__).parent / 'results' / 'logs'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Merge all N-MNIST TensorBoard files
tb_files = sorted(LOGS_DIR.glob('events.out.tfevents*'))
print(f"📂 Found {len(tb_files)} N-MNIST log files")

all_data = {
    'train/loss': {}, 'val/loss': {},
    'train/accuracy': {}, 'val/accuracy': {},
    'train/spike_rate': {}, 'val/spike_rate': {},
    'val/spike_efficiency': {},
}

for f in tb_files:
    ea = event_accumulator.EventAccumulator(str(f), size_guidance={'scalars': 0})
    ea.Reload()
    for tag in all_data:
        if tag in ea.Tags().get('scalars', []):
            for e in ea.Scalars(tag):
                all_data[tag][e.step] = e.value

# Convert to sorted lists
def sorted_vals(d):
    steps = sorted(d.keys())
    return steps, [d[s] for s in steps]

epochs, train_loss = sorted_vals(all_data['train/loss'])
_, val_loss = sorted_vals(all_data['val/loss'])
_, train_acc = sorted_vals(all_data['train/accuracy'])
_, val_acc = sorted_vals(all_data['val/accuracy'])
_, train_spikes = sorted_vals(all_data['train/spike_rate'])
_, val_spikes = sorted_vals(all_data['val/spike_rate'])
_, val_eff = sorted_vals(all_data.get('val/spike_efficiency', {}))

print(f"✅ Merged {len(epochs)} epochs from {len(tb_files)} log files")
print(f"🎯 Best val acc: {max(val_acc):.2f}% (epoch {epochs[val_acc.index(max(val_acc))]})")
print(f"⚡ Lowest spike rate: {min(val_spikes):.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(epochs, train_loss, '#44aaff', linewidth=1.5, label='Train', alpha=0.7)
axes[0].plot(epochs, val_loss, 'cyan', linewidth=2, label='Val')
axes[0].set_title('Classification Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(epochs, val_acc, '#00ff88', linewidth=2, label='Val Accuracy')
best_idx = val_acc.index(max(val_acc))
axes[1].axhline(y=max(val_acc), color='#00ff88', linestyle='--', alpha=0.3)
axes[1].text(epochs[-1], max(val_acc), f'  {max(val_acc):.2f}%', color='#00ff88', fontsize=12, va='bottom', fontweight='bold')
axes[1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Spike rate
axes[2].plot(epochs, val_spikes, '#ff6600', linewidth=2, label='Val Spike Rate')
if val_eff:
    axes[2].plot(epochs, val_eff, '#ffaa44', linewidth=1.5, label='Val Spike Efficiency', linestyle='--')
axes[2].set_title('Spike Efficiency', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Rate')
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.text(0.5, 0.01,
         'N-MNIST SNN — Surrogate Gradients | Whisper Architecture | 95%+ Neurons Silent',
         ha='center', fontsize=12, color='#666688', fontweight='bold')

plt.tight_layout()
out = OUTPUT_DIR / '01_nmnist_results.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"\n📊 Saved: {out}")
plt.close()
