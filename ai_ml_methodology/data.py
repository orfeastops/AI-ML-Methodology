import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from tqdm.auto import tqdm

# Data settings
T_MAX = 60.0          # s
DT = 0.1              # s
TIMESTEPS = int(T_MAX / DT)  # 600
TRACK_DIM = 4         # R, vr, az, el
BATCH = 2_000         # Increased batch size
N_TRACKS = 50_000     # Scaled up dataset
NOISE_STD = np.array([5, 3, 2e-3, 2e-3])
DATA_DIR = "./radar_data"
MODEL_DIR = "./radar_models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Enhanced missile specs
MISSILE_SPECS = {
    0: dict(name="ATACMS", v0_mu=1700, v0_sigma=80, th_mu=45, th_sigma=3),
    1: dict(name="Iskander", v0_mu=2000, v0_sigma=100, th_mu=50, th_sigma=4),
    2: dict(name="GMLRS", v0_mu=1300, v0_sigma=60, th_mu=42, th_sigma=3),
}

# Physics helpers
g = 9.81

def ballistic_state(t, v0=1700., theta_deg=45.):
    theta = np.deg2rad(theta_deg)
    v0x, v0y = v0 * np.cos(theta), v0 * np.sin(theta)
    x = v0x * t
    y = v0y * t - 0.5 * g * t**2
    vx = np.full_like(t, v0x)
    vy = v0y - g * t
    return x, y, vx, vy

def range_doppler_snapshot(x, y, vx, vy):
    R = np.hypot(x, y) + 1e-6
    vr = (x * vx + y * vy) / R
    az = np.arctan2(x, y)
    el = np.arctan2(y, x)
    return np.stack([R, vr, az, el], axis=-1)

def simulate_track(v0, theta):
    t = np.arange(0, T_MAX, DT)
    x, y, vx, vy = ballistic_state(t, v0, theta)
    snap = range_doppler_snapshot(x, y, vx, vy)
    noise = np.random.normal(0, NOISE_STD, snap.shape)
    for i in range(1, len(noise)):
        # Introduce some temporal correlation to the noise
        noise[i] = 0.95 * noise[i-1] + 0.05 * noise[i]
    snap += noise
    if np.any(np.isnan(snap)) or np.any(np.isinf(snap)):
        print(f"Warning: Invalid values in track for v0={v0}, theta={theta}")
    return snap.astype(np.float32)


# Data generation#

def generate_all(n_tracks=N_TRACKS):
    print(f"Generating {n_tracks} tracks in batches of {BATCH}...")
    all_data = []
    all_labels = []
    all_targets = []

    for b_idx in tqdm(range(0, n_tracks, BATCH)):
        actual_batch_size = min(BATCH, n_tracks - b_idx)
        data, labels, targets = [], [], []

        for _ in range(actual_batch_size):
            m_id = np.random.choice(list(MISSILE_SPECS))
            s = MISSILE_SPECS[m_id]
            v0 = np.random.normal(s["v0_mu"], s["v0_sigma"])
            theta = np.random.normal(s["th_mu"], s["th_sigma"])
            trk = simulate_track(v0, theta)

            idx = np.arange(TIMESTEPS) + 5
            idx[idx >= TIMESTEPS] = TIMESTEPS - 1
            future_pos = trk[idx, 0]
            targets.append(future_pos)

            data.append(trk)
            labels.append(m_id)

        batch_num = b_idx // BATCH
        np.save(f"{DATA_DIR}/data_{batch_num:03d}.npy", np.stack(data))
        np.save(f"{DATA_DIR}/label_{batch_num:03d}.npy", np.array(labels))
        np.save(f"{DATA_DIR}/target_{batch_num:03d}.npy", np.stack(targets))

        all_data.extend(data)
        all_labels.extend(labels)
        all_targets.extend(targets)

    # Compute normalization statistics
    print("Computing normalization statistics...")
    all_data = np.stack(all_data)
    all_targets = np.stack(all_targets)

    data_mean = np.mean(all_data, axis=(0, 1))
    data_std = np.std(all_data, axis=(0, 1))
    target_mean = np.mean(all_targets)
    target_std = np.std(all_targets)

    norm_stats = {
        'data_mean': data_mean,
        'data_std': data_std,
        'target_mean': target_mean,
        'target_std': target_std
    }
    with open(f"{DATA_DIR}/norm_stats.pkl", 'wb') as f:
        pickle.dump(norm_stats, f)

    print(f"✅ Dataset generated: {len(all_data)} samples")
    print(f"Data mean: {data_mean}")
    print(f"Data std: {data_std}")
    print(f"Target mean: {target_mean:.2f}, std: {target_std:.2f}")

def ensure_data_exists(n_tracks=N_TRACKS):
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".npy")]
    norm_stats_exists = os.path.exists(f"{DATA_DIR}/norm_stats.pkl")

    # Force regeneration if the number of track files doesn't match expected
    expected_files = math.ceil(n_tracks / BATCH)
    if len(data_files) != expected_files or not norm_stats_exists:
        print(f"Data mismatch or not found. Regenerating {n_tracks} tracks...")
        # Clean up old data before regenerating
        for item in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, item))
        generate_all(n_tracks)
    else:
        print(f"Found {len(data_files)} data files, data is ready.")


def compute_normalization_stats():
    all_data, all_targets = [], []
    data_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".npy")])

    for file in tqdm(data_files, desc="Loading data for stats"):
        file_num = int(file.split('_')[1].split('.')[0])
        data = np.load(f"{DATA_DIR}/data_{file_num:03d}.npy")
        targets = np.load(f"{DATA_DIR}/target_{file_num:03d}.npy")
        all_data.append(data)
        all_targets.append(targets)

    all_data = np.concatenate(all_data, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    data_mean = np.mean(all_data, axis=(0, 1))
    data_std = np.std(all_data, axis=(0, 1))
    target_mean = np.mean(all_targets)
    target_std = np.std(all_targets)

    norm_stats = {
        'data_mean': data_mean,
        'data_std': data_std,
        'target_mean': target_mean,
        'target_std': target_std
    }
    with open(f"{DATA_DIR}/norm_stats.pkl", 'wb') as f:
        pickle.dump(norm_stats, f)


# Dataset
class RadarDS(Dataset):
    def __init__(self, split="train", normalize=True, n_tracks=N_TRACKS):
        all_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".npy")])
        if len(all_files) == 0:
            raise ValueError(f"No data files found in {DATA_DIR}.")

        n = len(all_files)
    
        split_idx = max(1, int(0.8 * n))
        if split == "train":
            self.data_files = all_files[:split_idx]
        else: # "val" or "test"
            self.data_files = all_files[split_idx:]

        self.file_sizes = [np.load(f"{DATA_DIR}/{f}", mmap_mode='r').shape[0] for f in self.data_files]
        self.cumulative_sizes = np.cumsum(self.file_sizes)
        self.total_samples = self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0

        print(f"{split} dataset: {self.total_samples} samples from {len(self.data_files)} files.")

        self.normalize = normalize
        if self.normalize:
            norm_stats_path = f"{DATA_DIR}/norm_stats.pkl"
            if os.path.exists(norm_stats_path):
                with open(norm_stats_path, 'rb') as f:
                    self.norm_stats = pickle.load(f)
            else:
                raise FileNotFoundError("norm_stats.pkl not found. Please generate data first.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset with size {self.total_samples}")

        
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        local_idx = idx
        if file_idx > 0:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        file_name = self.data_files[file_idx]
        file_num = int(file_name.split('_')[1].split('.')[0])
        data_path = f"{DATA_DIR}/data_{file_num:03d}.npy"
        label_path = f"{DATA_DIR}/label_{file_num:03d}.npy"
        target_path = f"{DATA_DIR}/target_{file_num:03d}.npy"

        data = np.load(data_path, mmap_mode='r')[local_idx].copy()
        label = np.load(label_path, mmap_mode='r')[local_idx]
        target = np.load(target_path, mmap_mode='r')[local_idx].copy()

        if self.normalize:
            data = (data - self.norm_stats['data_mean']) / (self.norm_stats['data_std'] + 1e-8)
            target = (target - self.norm_stats['target_mean']) / (self.norm_stats['target_std'] + 1e-8)

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long), torch.from_numpy(target)