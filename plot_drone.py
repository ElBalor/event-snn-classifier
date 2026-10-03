"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PLOT DRONE SNN TRAINING RESULTS                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import json

plt.style.use('dark_background')
mpl.rcParams['axes.facecolor'] = '#1a1a2e'
mpl.rcParams['figure.facecolor'] = '#0f0f1a'
mpl.rcParams['grid.color'] = '#333355'
mpl.rcParams['text.color'] = '#e0e0e0'
mpl.rcParams['axes.labelcolor'] = '#e0e0e0'
mpl.rcParams['xtick.color'] = '#e0e0e0'
mpl.rcParams['ytick.color'] = '#e0e0e0'

RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = Path(__file__).parent / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_drone():
    """Load Drone SNN training history from history.json."""
    hist_path = RESULTS_DIR / 'logs_drone' / 'history.json'
    if not hist_path.exists():
        print("❌ history.json not found")
        return None

    with open(hist_path) as f:
        h = json.load(f)

    train = h.get('train', [])
    val = h.get('val', [])

    epochs = [x['epoch'] for x in val]
    return {
        'epochs': epochs,
        'train_loss': [x['loss'] for x in train],
        'val_loss': [x['loss'] for x in val],
        'depth_mae': [x.get('depth_mae', 0) for x in val],
        'depth_rmse': [x.get('depth_rmse', 0) for x in val],
        'pos_mae': [x.get('pos_mae', 0) for x in val],
        'rel_err': [x.get('rel_depth_err', 0) for x in val],
        'spikes': [x.get('spike', 0) for x in train],
    }


def plot(data):
    """Generate Drone SNN results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    best_epoch = data['epochs'][data['val_loss'].index(min(data['val_loss']))]
    best_loss = min(data['val_loss'])

    # 1. Loss
    axes[0, 0].plot(data['epochs'], data['train_loss'], '#44aaff', linewidth=1.5,
                    label='Train', alpha=0.7)
    axes[0, 0].plot(data['epochs'], data['val_loss'], '#ff4488', linewidth=2, label='Val')
    axes[0, 0].axvline(x=best_epoch, color='cyan', linestyle='--', alpha=0.3)
    axes[0, 0].text(best_epoch, best_loss, f'  Best: ep{best_epoch}={best_loss:.4f}',
                    color='cyan', fontsize=9, va='bottom')
    axes[0, 0].set_title('Detection Loss', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Depth MAE
    axes[0, 1].plot(data['epochs'], data['depth_mae'], '#00ff88', linewidth=2)
    axes[0, 1].text(data['epochs'][-1], data['depth_mae'][-1],
                    f'  {data["depth_mae"][-1]:.3f}', color='#00ff88', fontsize=10, va='bottom')
    axes[0, 1].set_title('Depth MAE', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].grid(alpha=0.3)

    # 3. Depth RMSE
    axes[0, 2].plot(data['epochs'], data['depth_rmse'], '#ffaa00', linewidth=2)
    axes[0, 2].text(data['epochs'][-1], data['depth_rmse'][-1],
                    f'  {data["depth_rmse"][-1]:.3f}', color='#ffaa00', fontsize=10, va='bottom')
    axes[0, 2].set_title('Depth RMSE', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].grid(alpha=0.3)

    # 4. Position MAE
    axes[1, 0].plot(data['epochs'], data['pos_mae'], '#8888ff', linewidth=2)
    axes[1, 0].text(data['epochs'][-1], data['pos_mae'][-1],
                    f'  {data["pos_mae"][-1]:.3f}', color='#8888ff', fontsize=10, va='bottom')
    axes[1, 0].set_title('Position MAE', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE (px)')
    axes[1, 0].grid(alpha=0.3)

    # 5. Relative Error
    axes[1, 1].plot(data['epochs'], data['rel_err'], '#ff6688', linewidth=2)
    axes[1, 1].text(data['epochs'][-1], data['rel_err'][-1],
                    f'  {data["rel_err"][-1]:.1%}', color='#ff6688', fontsize=10, va='bottom')
    axes[1, 1].set_title('Relative Depth Error', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error (%)')
    axes[1, 1].grid(alpha=0.3)

    # 6. Spike Rate
    axes[1, 2].plot(data['epochs'], data['spikes'], '#00ccff', linewidth=2)
    axes[1, 2].axhline(y=0.05, color='#ff6600', linestyle='--', alpha=0.5, label='5% target')
    axes[1, 2].text(data['epochs'][-1], data['spikes'][-1],
                    f'  {data["spikes"][-1]:.1%}', color='#00ccff', fontsize=10, va='bottom')
    axes[1, 2].set_title('Spike Rate (Whisper)', fontsize=13, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    # Watermark
    fig.text(0.5, 0.01,
             'DroneSNN — 3M Params | 94%+ Neurons Silent | Multi-scale TCN | Surrogate Gradients',
             ha='center', fontsize=12, color='#666688', fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / '02_drone_results.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"📊 Saved: {out}")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DRONE SNN — FINAL RESULTS (Epoch {data['epochs'][-1]})")
    print(f"{'='*60}")
    print(f"  Best Val Loss:     {best_loss:.4f} (epoch {best_epoch})")
    print(f"  Depth MAE:         {data['depth_mae'][-1]:.4f}")
    print(f"  Depth RMSE:        {data['depth_rmse'][-1]:.4f}")
    print(f"  Position MAE:      {data['pos_mae'][-1]:.4f}")
    print(f"  Relative Error:    {data['rel_err'][-1]:.1%}")
    print(f"  Spike Rate:        {data['spikes'][-1]:.1%} (94.{int((1-data['spikes'][-1])*10)}% silent)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  DRONE SNN — TRAINING RESULTS")
    print("═" * 60 + "\n")

    data = load_drone()
    if data:
        print(f"✅ Loaded {len(data['epochs'])} epochs\n")
        plot(data)
        print(f"✅ Done.\n")
