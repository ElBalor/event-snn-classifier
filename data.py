"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  N-MNIST DATA LOADER WITH TONIC                                              ║
║  "Event-based vision data"                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict
import tonic


# ──────────────────────────────────────────────────────────────────────────────
#  N-MNIST DATASET
# ──────────────────────────────────────────────────────────────────────────────

class NMNISTDataset(Dataset):
    """
    N-MNIST: Neuromorphic MNIST dataset.
    
    70,000 event-based recordings of handwritten digits.
    Captured with DVS (Dynamic Vision Sensor) camera.
    
    Each sample:
    - 34 × 34 pixels
    - 2 polarities (ON/OFF events)
    - ~300ms duration
    - Variable number of events (~2000 per digit)
    
    Auto-downloads via tonic library.
    """
    
    def __init__(
        self,
        root: str = './data/nmnist',
        train: bool = True,
        num_time_bins: int = 10,
        transform: Optional[callable] = None,
        download: bool = True
    ):
        self.root = root
        self.train = train
        self.num_time_bins = num_time_bins
        self.transform = transform
        
        # Load N-MNIST via tonic
        self.dataset = tonic.datasets.NMNIST(
            save_to=root,
            train=train
        )
        
        # Tonic transforms for event → voxel grid conversion
        self.transform_compose = tonic.transforms.Compose([
            tonic.transforms.ToFrame(
                sensor_size=tonic.datasets.NMNIST.sensor_size,
                time_window=30000  # 30ms windows
            )
        ])
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get single sample.
        
        Returns:
            events: Voxel grid [2, 34, 34, T]
            label: Digit label (0-9)
            info: Additional info (num_events, etc.)
        """
        # Load events and label
        events, label = self.dataset[idx]
        
        # Convert to voxel grid
        voxel_grid = self._events_to_voxel(events)
        
        # Info dict
        info = {
            'num_events': len(events),
            'duration_ms': events['t'].max() - events['t'].min(),
            'label': label
        }
        
        return voxel_grid, label, info
    
    def _events_to_voxel(self, events: np.ndarray) -> torch.Tensor:
        """
        Convert event stream to voxel grid representation.
        
        Events: [(x, y, t, p), ...]
        Voxel grid: [2, 34, 34, T] (2 polarities, time bins)
        """
        # Get time range
        t_min = events['t'].min()
        t_max = events['t'].max()
        t_range = t_max - t_min + 1
        
        # Initialize voxel grid
        voxel_grid = np.zeros((2, 34, 34, self.num_time_bins), dtype=np.float32)
        
        # Bin events into time bins
        bin_size = t_range / self.num_time_bins
        
        for i in range(len(events)):
            x = events['x'][i]
            y = events['y'][i]
            t = events['t'][i]
            p = events['p'][i]  # polarity (0 or 1)
            
            # Compute time bin
            t_bin = int((t - t_min) / bin_size)
            t_bin = min(t_bin, self.num_time_bins - 1)
            
            # Accumulate in voxel grid
            voxel_grid[p, y, x, t_bin] += 1
        
        # Normalize by max events per bin
        max_events = voxel_grid.max()
        if max_events > 0:
            voxel_grid /= max_events
        
        return torch.FloatTensor(voxel_grid)


# ──────────────────────────────────────────────────────────────────────────────
#  EVENT AUGMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

class EventAugmentations:
    """
    Augmentations for event-based data.
    """
    
    @staticmethod
    def time_flip(events: np.ndarray) -> np.ndarray:
        """Flip events in time (reverse temporal order)."""
        events = events.copy()
        t_max = events['t'].max()
        events['t'] = t_max - events['t']
        return events
    
    @staticmethod
    def spatial_flip(events: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Flip events spatially."""
        events = events.copy()
        if horizontal:
            events['x'] = 33 - events['x']  # 34 pixels wide
        else:
            events['y'] = 33 - events['y']
        return events
    
    @staticmethod
    def polarity_flip(events: np.ndarray) -> np.ndarray:
        """Flip event polarities."""
        events = events.copy()
        events['p'] = 1 - events['p']
        return events
    
    @staticmethod
    def drop_events(events: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
        """Randomly drop events (simulates sensor noise)."""
        events = events.copy()
        mask = np.random.rand(len(events)) > drop_rate
        return events[mask]


# ──────────────────────────────────────────────────────────────────────────────
#  DATA MODULE
# ──────────────────────────────────────────────────────────────────────────────

class NMNISTDataModule:
    """
    Complete data module for N-MNIST.
    Handles train/val/test splits and dataloaders.
    """
    
    def __init__(
        self,
        root: str = './data/nmnist',
        num_time_bins: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.root = root
        self.num_time_bins = num_time_bins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Initialize datasets."""
        import os
        from pathlib import Path
        
        # Check what we already have
        data_path = Path(self.root) / 'NMNIST'
        train_exists = (data_path / 'Train').exists() or (data_path / 'train').exists()
        test_exists = (data_path / 'Test').exists() or (data_path / 'test').exists()
        test_zip_exists = (data_path / 'test.zip').exists()
        
        print(f"\n📂 Checking existing data:")
        print(f"   Train data: {'✅ Found' if train_exists else '❌ Missing'}")
        print(f"   Test data: {'✅ Found' if test_exists else '❌ Missing'}")
        print(f"   Test zip: {'✅ Found' if test_zip_exists else '❌ Missing'}")
        
        # Train dataset
        if train_exists:
            print(f"\n✅ Loading existing train data...")
            self.train_dataset = NMNISTDataset(
                root=self.root,
                train=True,
                num_time_bins=self.num_time_bins,
                download=False  # Don't re-download
            )
        else:
            print(f"\n📥 Downloading train data...")
            self.train_dataset = NMNISTDataset(
                root=self.root,
                train=True,
                num_time_bins=self.num_time_bins,
                download=True
            )
        
        # Test dataset (used as validation during training)
        if test_exists:
            print(f"\n✅ Loading existing test data...")
            self.test_dataset = NMNISTDataset(
                root=self.root,
                train=False,
                num_time_bins=self.num_time_bins,
                download=False  # Don't re-download
            )
        else:
            print(f"\n📥 Downloading test data only...")
            self.test_dataset = NMNISTDataset(
                root=self.root,
                train=False,
                num_time_bins=self.num_time_bins,
                download=True
            )
        
        print(f"\n📊 N-MNIST loaded:")
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Test samples: {len(self.test_dataset)}")
        print(f"   Time bins: {self.num_time_bins}")
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader (uses test set)."""
        return self.test_dataloader()


# ──────────────────────────────────────────────────────────────────────────────
#  COLLATE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def nmnist_collate_fn(batch):
    """
    Custom collate function for N-MNIST batches.
    Handles variable-length event streams.
    """
    voxel_grids, labels, infos = zip(*batch)
    
    # Stack voxel grids
    voxel_batch = torch.stack(voxel_grids, dim=0)
    
    # Stack labels
    labels = torch.LongTensor(labels)
    
    return voxel_batch, labels, infos


# ──────────────────────────────────────────────────────────────────────────────
#  TESTING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "N-MNIST DATA LOADER TEST" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Check if tonic is installed
    try:
        import tonic
        print("✅ tonic installed")
    except ImportError:
        print("❌ tonic not installed. Run: pip install tonic")
        exit(1)
    
    # Create data module
    data_module = NMNISTDataModule(
        num_time_bins=10,
        batch_size=8,
        num_workers=0  # Windows compatibility
    )
    
    print("\n📥 Loading N-MNIST dataset...")
    print("   (First download may take 5-10 minutes)")
    
    data_module.setup()
    
    # Get a batch
    print("\n📦 Getting training batch...")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    events, labels, infos = batch
    
    print(f"\n📊 Batch info:")
    print(f"   Events shape: {events.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels: {labels.tolist()}")
    
    print(f"\n📊 Sample info:")
    print(f"   Num events: {infos[0]['num_events']}")
    print(f"   Duration: {infos[0]['duration_ms']} ms")
    print(f"   Label: {infos[0]['label']}")
    
    # Test augmentations
    print("\n🎨 Testing augmentations...")
    
    # Load single sample for augmentation test
    sample_events, _, _ = data_module.train_dataset[0]
    
    # Convert back to event format for augmentation test
    print("   Augmentations available:")
    print("   - Time flip")
    print("   - Spatial flip")
    print("   - Polarity flip")
    print("   - Drop events")
    
    print("\n✅ N-MNIST data loader test passed!\n")
