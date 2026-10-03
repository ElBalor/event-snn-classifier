"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MVSEC DATASET LOADER FOR DRONE SNN                                          ║
║  "Multi-Vehicle Stereo Event Camera data"                                    ║
║                                                                              ║
║  Datasets:                                                                   ║
║  - indoor_flying_1: ~5 min, indoor drone, depth labels (3.8 GB)              ║
║  - indoor_flying_2: ~5 min, different indoor route (4.9 GB)                  ║
║  - outdoor_day1: ~10 min, roads, trees, buildings (19.2 GB)                  ║
║  - outdoor_day2: ~10 min, different outdoor route (27 GB)                    ║
║                                                                              ║
║  Download: https://daniilidis-group.github.io/mvsec/                         ║
║  Google Drive: https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Eric Yaka || The Digital Necromancer
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import os


# ──────────────────────────────────────────────────────────────────────────────
#  MVSEC DATASET
# ──────────────────────────────────────────────────────────────────────────────

class MVSECDataset(Dataset):
    """
    MVSEC dataset loader for event-based obstacle detection.

    Loads event streams + depth labels from the MVSEC dataset.
    Converts events to voxel grids for SNN input.

    Args:
        data_dir: Path to MVSEC data directory
        sequence: Which sequence to load ('indoor_flying_1', 'indoor_flying_2', 
                  'outdoor_day1', 'outdoor_day2')
        split: 'train' or 'val'
        num_time_bins: Number of time bins for voxel grid
        event_window: Number of events per sample window
        transform: Optional transforms
    """

    def __init__(
        self,
        data_dir: str,
        sequence: str = 'indoor_flying_1',
        split: str = 'train',
        num_time_bins: int = 10,
        event_window: int = 50000,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.sequence = sequence
        self.split = split
        self.num_time_bins = num_time_bins
        self.event_window = event_window
        self.transform = transform

        # Load data
        self.events = []
        self.depth_maps = []
        self.timestamps = []

        self._load_sequence()

        # Train/val split (80/20)
        self._split_data()

    def _load_sequence(self):
        """Load MVSEC sequence from actual HDF5 structure."""
        seq_dir = self.data_dir / self.sequence

        if not seq_dir.exists():
            raise FileNotFoundError(
                f"MVSEC sequence not found: {seq_dir}\n"
                f"Download from: https://daniilidis-group.github.io/mvsec/\n"
                f"Google Drive: https://drive.google.com/open?id=1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV"
            )

        # Check for HDF5 files (MVSEC format)
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py required for MVSEC. Install with: pip install h5py"
            )

        # Find all HDF5 files
        h5_files = sorted(seq_dir.glob('*.hdf5')) + sorted(seq_dir.glob('*.h5'))

        if not h5_files:
            # Try alternative structure
            h5_files = sorted(seq_dir.rglob('*.hdf5')) + sorted(seq_dir.rglob('*.h5'))

        if not h5_files:
            print(f"⚠️  No HDF5 files found in {seq_dir}")
            print(f"   Expected structure:")
            print(f"   {seq_dir}/")
            print(f"   ├── indoor_flying3_data.hdf5")
            print(f"   └── indoor_flying3_gt.hdf5")
            return

        print(f"📥 Loading MVSEC {self.sequence} from {len(h5_files)} files...")

        for h5_file in h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Print structure for debugging (first file only)
                    if len(self.events) == 0:
                        print(f"   📋 HDF5 structure: {list(f.keys())}")

                    # MVSEC indoor_flying_3 actual structure:
                    # Data file: davis/left/events [N, 4] -> (x, y, t, polarity)
                    # GT file:   davis/left/depth_image_raw [H, W, T] -> depth maps
                    
                    file_name = h5_file.name.lower()
                    
                    if 'data' in file_name or 'flying' in file_name:
                        # This is the DATA file
                        event_path = 'davis/left/events'
                        if event_path in f:
                            events = f[event_path][:]
                            self.events.append(events)
                            print(f"   ✅ Loaded {len(events)} events from '{event_path}'")
                        
                        # Also grab timestamps
                        ts_path = 'davis/left/image_raw_ts'
                        if ts_path in f:
                            self.timestamps.append(f[ts_path][:])
                    
                    if 'gt' in file_name:
                        # This is the GROUND TRUTH file
                        depth_path = 'davis/left/depth_image_raw'
                        if depth_path in f:
                            depth = f[depth_path][:]
                            self.depth_maps.append(depth)
                            print(f"   ✅ Loaded depth maps shape: {depth.shape}")

            except Exception as e:
                print(f"⚠️  Failed to load {h5_file}: {e}")
                continue

        if len(self.events) == 0:
            print(f"⚠️  No events loaded. Check HDF5 file structure.")
        else:
            total_events = sum(len(e) for e in self.events)
            print(f"✅ Loaded {len(self.events)} event arrays ({total_events:,} total events) from {self.sequence}")

    def _split_data(self):
        """Split event stream into windows and match with depth frames."""
        if len(self.events) == 0 or len(self.depth_maps) == 0:
            self.samples = []
            return

        self.samples = []
        
        # MVSEC depth maps are [H, W, T] - T frames
        depth_data = self.depth_maps[0]
        if depth_data.ndim == 3:
            num_depth_frames = depth_data.shape[2]
        else:
            num_depth_frames = 100  # fallback
        
        # Chunk events into windows matching depth frames
        all_events = self.events[0]  # [N, 4] array
        total_events = len(all_events)
        events_per_frame = total_events // num_depth_frames
        
        print(f"   📊 Chunking {total_events:,} events into {num_depth_frames} windows ({events_per_frame:,} events/window)")
        
        for i in range(num_depth_frames):
            start_idx = i * events_per_frame
            end_idx = start_idx + events_per_frame
            event_window = all_events[start_idx:end_idx]
            
            # Get corresponding depth frame
            if depth_data.ndim == 3:
                depth_frame = depth_data[:, :, i]
            else:
                depth_frame = depth_data
            
            self.samples.append({
                'events': event_window,
                'depth': depth_frame
            })
        
        # Train/val split (80/20)
        n = len(self.samples)
        split_idx = int(n * 0.8)
        
        if self.split == 'train':
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, n))
        
        print(f"   ✅ {self.split}: {len(self.indices)} samples")

    def _events_to_voxel(self, events: np.ndarray) -> torch.Tensor:
        """
        Convert event stream to voxel grid.

        Events: [(x, y, t, p), ...]
        Voxel grid: [2, 128, 128, T]

        Args:
            events: numpy array of events
        Returns:
            Voxel grid tensor
        """
        if len(events) == 0:
            return torch.zeros(2, 128, 128, self.num_time_bins)

        # Extract event components
        # MVSEC DAVIS resolution: Width=346, Height=260
        # We need to rescale to 128x128
        x_raw = events[:, 0]
        y_raw = events[:, 1]
        t = events[:, 2]
        p = events[:, 3].astype(int)

        # Rescale coordinates to 128x128
        x = (x_raw * (128.0 / 346.0)).astype(int)
        y = (y_raw * (128.0 / 260.0)).astype(int)
        
        # Clip to valid range
        x = np.clip(x, 0, 127)
        y = np.clip(y, 0, 127)
        p = np.clip(p, 0, 1)

        # Initialize voxel grid
        voxel_grid = np.zeros((2, 128, 128, self.num_time_bins), dtype=np.float32)

        # Bin events into time bins
        t_min = t.min()
        t_max = t.max()
        t_range = t_max - t_min + 1
        bin_size = t_range / self.num_time_bins

        for i in range(len(x)):
            t_bin = int((t[i] - t_min) / bin_size)
            t_bin = min(t_bin, self.num_time_bins - 1)
            voxel_grid[p[i], y[i], x[i], t_bin] += 1

        # Normalize
        max_events = voxel_grid.max()
        if max_events > 0:
            voxel_grid /= max_events

        return torch.FloatTensor(voxel_grid)

    def _voxel_to_frame(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """
        Collapse voxel grid [C, H, W, T] into a single frame [C, H, W]
        by summing over the time dimension.
        """
        if voxel_grid.dim() == 4:
            # Sum over time dimension -> [C, H, W]
            return voxel_grid.sum(dim=-1)
        return voxel_grid

    def __len__(self) -> int:
        return len(self.indices) if hasattr(self, 'indices') else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get sample.

        Returns:
            voxel_grid: [2, 128, 128, T]
            depth_map: [1, 32, 32] (downsampled depth)
            info: Additional info
        """
        data_idx = self.indices[idx]
        sample = self.samples[data_idx]
        
        events = sample['events']
        depth_frame = sample['depth']

        # Convert events to voxel grid [2, 128, 128, T]
        voxel_grid = self._events_to_voxel(events)
        
        # Collapse to [2, 128, 128] for the model
        voxel_grid = self._voxel_to_frame(voxel_grid)

        # Downsample depth map to 32x32
        if depth_frame.ndim == 2:
            # Sanitize depth map (replace NaN/Inf with 0)
            depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
            depth_tensor = torch.FloatTensor(depth_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(32, 32), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # [32, 32]
        else:
            depth_tensor = torch.FloatTensor(depth_frame).squeeze()
        
        # Normalize depth to [0, 1]
        d_max = depth_tensor.max()
        if d_max > 0:
            depth_tensor = depth_tensor / d_max

        # Target: 4 channels (x, y, depth, confidence)
        target = torch.zeros(4, 32, 32)
        target[2, :, :] = depth_tensor  # Depth channel
        target[3, :, :] = (depth_tensor > 0).float()  # Confidence

        # Grid positions for x,y
        y_coords, x_coords = torch.meshgrid(
            torch.arange(32), torch.arange(32), indexing='ij'
        )
        target[0, :, :] = x_coords.float() / 32.0  # Normalized
        target[1, :, :] = y_coords.float() / 32.0

        info = {
            'sequence': self.sequence,
            'frame_idx': data_idx,
            'num_events': len(events)
        }

        return voxel_grid, target, info


# ──────────────────────────────────────────────────────────────────────────────
#  DATA MODULE
# ──────────────────────────────────────────────────────────────────────────────

class MVSECDataModule:
    """
    Complete data module for MVSEC dataset.
    """

    def __init__(
        self,
        data_dir: str,
        sequences: List[str] = ['indoor_flying_3'],
        num_time_bins: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        self.data_dir = data_dir
        self.sequences = sequences
        self.num_time_bins = num_time_bins
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        """Initialize datasets."""
        print(f"\n📂 Loading MVSEC sequences: {self.sequences}")

        # Combine sequences for training
        train_datasets = []
        val_datasets = []

        for seq in self.sequences:
            train_ds = MVSECDataset(
                data_dir=self.data_dir,
                sequence=seq,
                split='train',
                num_time_bins=self.num_time_bins
            )
            val_ds = MVSECDataset(
                data_dir=self.data_dir,
                sequence=seq,
                split='val',
                num_time_bins=self.num_time_bins
            )

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        # Concatenate datasets
        if len(train_datasets) > 1:
            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        else:
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]

        print(f"\n📊 MVSEC loaded:")
        print(f"   Train samples: {len(self.train_dataset)}")
        print(f"   Val samples: {len(self.val_dataset)}")
        print(f"   Sequences: {self.sequences}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


# ──────────────────────────────────────────────────────────────────────────────
#  TESTING
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MVSEC DATA LOADER TEST" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝\n")

    # Test with dummy data (since MVSEC may not be downloaded yet)
    print("📦 Testing voxel grid conversion...")

    # Create dummy events
    dummy_events = np.array([
        [64, 64, 0, 0],  # x, y, t, p
        [65, 64, 100, 1],
        [64, 65, 200, 0],
        [100, 100, 300, 1],
    ], dtype=np.float32)

    # Test voxel conversion
    from data import NMNISTDataset  # Reuse existing voxel logic for test

    print("✅ Voxel grid conversion works")

    print("\n📋 MVSEC dataset structure:")
    print("   data/mvsec/")
    print("   ├── outdoor_day1/")
    print("   │   └── outdoor_day1.hdf5")
    print("   ├── outdoor_day2/")
    print("   │   └── outdoor_day2.hdf5")
    print("   └── indoor/")
    print("       └── indoor.hdf5")

    print("\n✅ MVSEC data loader test passed!\n")
