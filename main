!pip install -q pytorch-lightning torchmetrics onnx onnxruntime onnxruntime-gpu

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import Accuracy
from tqdm.auto import tqdm 
from sklearn.preprocessing import StandardScaler
import pickle
import time
import onnx
import onnxruntime as ort
from pathlib import Path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
try:
  
    import onnxruntime_gpu
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("ONNX Runtime will use CUDAExecutionProvider.")
except ImportError:
    providers = ['CPUExecutionProvider']
    print("ONNX Runtime will use CPUExecutionProvider.")



pl.seed_everything(42)

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

def generate_all():
    print(f"Generating {N_TRACKS} tracks in batches of {BATCH}...")
    all_data = []
    all_labels = []
    all_targets = []

    for b_idx in tqdm(range(0, N_TRACKS, BATCH)):
        actual_batch_size = min(BATCH, N_TRACKS - b_idx)
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

def ensure_data_exists():
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".npy")]
    norm_stats_exists = os.path.exists(f"{DATA_DIR}/norm_stats.pkl")

    # Force regeneration if the number of track files doesn't match expected
    expected_files = math.ceil(N_TRACKS / BATCH)
    if len(data_files) != expected_files or not norm_stats_exists:
        print(f"Data mismatch or not found. Regenerating {N_TRACKS} tracks...")
        # Clean up old data before regenerating
        for item in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, item))
        generate_all()
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

# Call data generation/verification at the start
ensure_data_exists()


# Dataset
class RadarDS(Dataset):
    def __init__(self, split="train", normalize=True):
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


# Model with MC-Dropout

class MCDropoutNet(pl.LightningModule):
    def __init__(self, dropout_rate=0.3, hidden_dim=256, mc_samples=20):
        super().__init__()
        self.save_hyperparameters()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate

        # CNN backbone
        self.conv = nn.Sequential(
            nn.Conv1d(TRACK_DIM, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout_rate), # Using standard nn.Dropout which works on any tensor shape
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )

        # RNN
        self.rnn = nn.LSTM(128, hidden_dim // 2, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout_rate)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, dropout=dropout_rate)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, TIMESTEPS)
        )

        self.acc = Accuracy(task="multiclass", num_classes=3)
        self.cls_weight = nn.Parameter(torch.tensor(1.0))
        self.reg_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
     
        x = x.transpose(1, 2)  # (B, 4, 600) -> (B, 600, 4)
        x = self.conv(x)       # (B, 128, 600)
        x = x.transpose(1, 2)  # (B, 600, 128)

        rnn_out, (h, _) = self.rnn(x)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        pooled = torch.mean(attn_out, dim=1)

        cls_out = self.cls_head(pooled)
        reg_out = self.reg_head(pooled)

        return cls_out, reg_out

    def mc_predict(self, x, n_samples=None):
        if n_samples is None:
            n_samples = self.mc_samples

        self.train()  
        cls_predictions = []
        reg_predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                cls_out, reg_out = self.forward(x)
                cls_predictions.append(F.softmax(cls_out, dim=-1))
                reg_predictions.append(reg_out)

        self.eval()  # Disable dropout layers for normal inference later

        cls_stack = torch.stack(cls_predictions)
        reg_stack = torch.stack(reg_predictions)

        cls_mean = torch.mean(cls_stack, dim=0)
        cls_std = torch.std(cls_stack, dim=0)
        cls_entropy = -torch.sum(cls_mean * torch.log(cls_mean + 1e-9), dim=-1)

        reg_mean = torch.mean(reg_stack, dim=0)
        reg_std = torch.std(reg_stack, dim=0)

        return {
            'cls_mean': cls_mean, 'cls_std': cls_std, 'cls_entropy': cls_entropy,
            'reg_mean': reg_mean, 'reg_std': reg_std
        }

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat, z_hat = self(x)

        cls_loss = F.cross_entropy(y_hat, y)
        reg_loss = F.mse_loss(z_hat, z)

        total_loss = torch.abs(self.cls_weight) * cls_loss + torch.abs(self.reg_weight) * reg_loss
        self.log_dict({
            "train_acc": self.acc(y_hat, y), "train_loss": total_loss,
            "train_cls_loss": cls_loss, "train_reg_loss": reg_loss
        }, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat, z_hat = self(x)

        cls_loss = F.cross_entropy(y_hat, y)
        reg_loss = F.mse_loss(z_hat, z)
        total_loss = torch.abs(self.cls_weight) * cls_loss + torch.abs(self.reg_weight) * reg_loss
        self.log_dict({
            "val_acc": self.acc(y_hat, y), "val_loss": total_loss,
            "val_cls_loss": cls_loss, "val_reg_loss": reg_loss
        }, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return { "optimizer": optimizer, "lr_scheduler": { "scheduler": scheduler, "monitor": "val_loss" } }


# ONNX Export & Optimization
def export_to_onnx(model, save_path="radar_model.onnx"):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, TIMESTEPS, TRACK_DIM, device=device)

    try:
        torch.onnx.export(
            model, dummy_input, save_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['input'], output_names=['classification', 'regression'],
            dynamic_axes={'input': {0: 'batch_size'}, 'classification': {0: 'batch_size'}, 'regression': {0: 'batch_size'}}
        )
        print(f"✅ Model exported to {save_path}")
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise
    return save_path

def benchmark_onnx_model(onnx_path, test_loader, n_runs=100):
    print("🚀 Benchmarking ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=providers)
    x_sample, _, _ = next(iter(test_loader))
    x_sample = x_sample.numpy()

    # Warmup
    for _ in range(10):
        _ = session.run(None, {'input': x_sample})

    times = []
    for _ in tqdm(range(n_runs), desc="ONNX Benchmark"):
        start_time = time.perf_counter()
        _ = session.run(None, {'input': x_sample})
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    print(f"✅ ONNX Inference (batch={x_sample.shape[0]}): {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Throughput: {x_sample.shape[0] / (avg_time / 1000):.1f} samples/sec")
    return avg_time, std_time

# Comprehensive Benchmarking
def comprehensive_benchmark(model, test_loader, save_results=True):
    print("🔍 Running comprehensive benchmark...")
    model.eval()
    device = next(model.parameters()).device

    all_cls_preds, all_reg_preds, all_cls_true, all_reg_true = [], [], [], []
    all_cls_uncertainty, all_reg_uncertainty = [], []
    pytorch_times, mc_times = [], []

    with open(f"{DATA_DIR}/norm_stats.pkl", 'rb') as f:
        norm_stats = pickle.load(f)

    with torch.no_grad():
        for x, y_true, z_true in tqdm(test_loader, desc="Comprehensive Benchmark"):
            x, y_true, z_true = x.to(device), y_true.to(device), z_true.to(device)

            start_time = time.perf_counter()
            y_pred, z_pred = model(x)
            pytorch_times.append(time.perf_counter() - start_time)

            start_time = time.perf_counter()
            mc_results = model.mc_predict(x)
            mc_times.append(time.perf_counter() - start_time)

            all_cls_preds.append(torch.argmax(y_pred, dim=-1).cpu())
            all_reg_preds.append(z_pred.cpu())
            all_cls_true.append(y_true.cpu())
            all_reg_true.append(z_true.cpu())
            all_cls_uncertainty.append(mc_results['cls_entropy'].cpu())
            all_reg_uncertainty.append(torch.mean(mc_results['reg_std'], dim=-1).cpu())

    cls_preds, reg_preds = torch.cat(all_cls_preds), torch.cat(all_reg_preds)
    cls_true, reg_true = torch.cat(all_cls_true), torch.cat(all_reg_true)
    cls_uncertainty, reg_uncertainty = torch.cat(all_cls_uncertainty), torch.cat(all_reg_uncertainty)

    # De-normalize regression results for interpretability
    reg_preds_denorm = reg_preds * norm_stats['target_std'] + norm_stats['target_mean']
    reg_true_denorm = reg_true * norm_stats['target_std'] + norm_stats['target_mean']

    cls_accuracy = (cls_preds == cls_true).float().mean().item()
    reg_rmse = torch.sqrt(torch.mean((reg_preds_denorm - reg_true_denorm) ** 2)).item()
    reg_mae = torch.mean(torch.abs(reg_preds_denorm - reg_true_denorm)).item()

    batch_size = test_loader.batch_size
    avg_pytorch_time = np.mean(pytorch_times) * 1000
    avg_mc_time = np.mean(mc_times) * 1000

    results = {
        'classification_accuracy': cls_accuracy, 'regression_rmse': reg_rmse, 'regression_mae': reg_mae,
        'pytorch_inference_ms_per_batch': avg_pytorch_time, 'mc_dropout_inference_ms_per_batch': avg_mc_time,
        'avg_classification_uncertainty': cls_uncertainty.mean().item(),
        'avg_regression_uncertainty': reg_uncertainty.mean().item(),
    }

    print("\n📊 BENCHMARK RESULTS:"); print("="*50)
    print(f"Classification Accuracy: {cls_accuracy:.4f}")
    print(f"Regression RMSE (denorm): {reg_rmse:.2f} meters")
    print(f"Regression MAE (denorm):  {reg_mae:.2f} meters")
    print(f"PyTorch Inference (batch={batch_size}): {avg_pytorch_time:.2f} ms")
    print(f"MC-Dropout Inference (batch={batch_size}): {avg_mc_time:.2f} ms")
    print(f"Cls Uncertainty (entropy): {results['avg_classification_uncertainty']:.4f}")
    print(f"Reg Uncertainty (mean std): {results['avg_regression_uncertainty']:.4f}")
    print("="*50)

    if save_results:
        with open(f"{MODEL_DIR}/benchmark_results.pkl", 'wb') as f: pickle.dump(results, f)

    return results

# Noise Robustness Testing
def test_noise_robustness(model, base_noise_std=NOISE_STD, n_samples=500):
    print("🔬 Testing noise robustness...")
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = {}

    device = next(model.parameters()).device
    model.eval()

    with open(f"{DATA_DIR}/norm_stats.pkl", "rb") as f: norm = pickle.load(f)
    data_mean = torch.tensor(norm['data_mean'], dtype=torch.float32, device=device)
    data_std = torch.tensor(norm['data_std'], dtype=torch.float32, device=device)

    for mult in noise_levels:
        print(f"  ➜   Testing at {mult}× noise level")
        xs, ys = [], []

        for _ in range(n_samples):
            m_id = np.random.choice(list(MISSILE_SPECS))
            spec = MISSILE_SPECS[m_id]
            v0 = np.random.normal(spec["v0_mu"], spec["v0_sigma"])
            th = np.random.normal(spec["th_mu"], spec["th_sigma"])
            
            track = simulate_track(v0, th)
            # Add extra, uncorrelated noise for testing
            extra_noise = np.random.normal(0, base_noise_std * mult, track.shape)
            track_noisy = track + extra_noise.astype(np.float32)

            track_tensor = torch.from_numpy(track_noisy).float()
            track_norm = (track_tensor - data_mean.cpu()) / (data_std.cpu() + 1e-8)
            xs.append(track_norm)
            ys.append(m_id)

        X = torch.stack(xs).to(device)
        Y = torch.tensor(ys, dtype=torch.long, device=device)
        
        with torch.no_grad():
            cls_out, _ = model(X)
            acc = (cls_out.argmax(-1) == Y).float().mean().item()
        
        results[mult] = {"acc": acc}
        print(f"      Accuracy = {acc:.3f}")
        
    return results


# Training & Evaluation Pipeline
def train_and_evaluate():
    print("🚀 Starting enhanced training pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    train_loader = DataLoader(RadarDS("train"), batch_size=64, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
    val_loader = DataLoader(RadarDS("val"), batch_size=128, num_workers=os.cpu_count(), pin_memory=True)

    model = MCDropoutNet(dropout_rate=0.25, hidden_dim=256, mc_samples=25)

    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", dirpath=f"{MODEL_DIR}/checkpoints/", filename="radar-best", save_top_k=1, mode="max"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=25, accelerator="auto", devices="auto",
        precision="16-mixed",
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        log_every_n_steps=50, gradient_clip_val=1.0,
    )

    print("🔥 Training model...")
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("⚠️ No best model saved. Using last model state for evaluation.")
        model.to(device) # Ensure model is on the correct device
    else:
        print(f"✅ Best model loaded from: {best_model_path}")
        model = MCDropoutNet.load_from_checkpoint(best_model_path).to(device)
    
    onnx_path = f"{MODEL_DIR}/radar_model.onnx"
    export_to_onnx(model, onnx_path)

    benchmark_results = comprehensive_benchmark(model, val_loader)
    onnx_avg_time, _ = benchmark_onnx_model(onnx_path, val_loader)
    noise_results = test_noise_robustness(model)

    print("\n🎯 FINAL SUMMARY:")
    print("=" * 60)
    print(f"✅ Model trained with {N_TRACKS:,} samples.")
    print(f"✅ Best Validation Accuracy: {benchmark_results['classification_accuracy']:.4f}")
    print(f"✅ Final Regression RMSE: {benchmark_results['regression_rmse']:.2f} meters")
    print(f"✅ PyTorch inference (batch={val_loader.batch_size}): {benchmark_results['pytorch_inference_ms_per_batch']:.2f} ms")
    print(f"✅ MC-Dropout inference (batch={val_loader.batch_size}): {benchmark_results['mc_dropout_inference_ms_per_batch']:.2f} ms")
    print(f"✅ ONNX inference (batch={val_loader.batch_size}): {onnx_avg_time:.2f} ms")
    print("✅ Noise Robustness (Accuracy vs. Noise Multiplier):")
    for level, res in noise_results.items():
        print(f"   - {level}x Noise: {res['acc']:.4f}")
    print(f"✅ Best model checkpoint and ONNX model saved in '{MODEL_DIR}'")
    print("=" * 60)

# Execute the main function
train_and_evaluate()
