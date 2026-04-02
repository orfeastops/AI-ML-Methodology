import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ai_ml_methodology.data import RadarDS, ensure_data_exists
from ai_ml_methodology.model import MCDropoutNet
from ai_ml_methodology.utils import comprehensive_benchmark, test_noise_robustness, export_to_onnx, benchmark_onnx_model

def train_and_evaluate(demo=False):
    print(" Starting training pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    worker_count = 0 if os.name == "nt" else max(1, min(4, os.cpu_count() or 1))
    pin_memory = device.type == "cuda"

    n_tracks = 4000 if demo else 50000
    ensure_data_exists(n_tracks)

    train_loader = DataLoader(
        RadarDS("train", n_tracks=n_tracks),
        batch_size=64,
        num_workers=worker_count,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        RadarDS("val", n_tracks=n_tracks),
        batch_size=128,
        num_workers=worker_count,
        pin_memory=pin_memory,
    )

    model = MCDropoutNet(dropout_rate=0.3, hidden_dim=256, mc_samples=20, learning_rate=1e-3)

    has_val = val_loader is not None and len(val_loader.dataset) > 0
    early_stop_monitor = "val_loss" if has_val else "train_loss"
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=8, min_delta=0.001, mode="min")
    checkpoint_monitor = "val_acc" if has_val else "train_acc"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=checkpoint_monitor, dirpath="./radar_models/checkpoints/", filename="radar-best", save_top_k=1, mode="max"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if device.type == "cuda" else "32-true",
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    print(" Training model...")
    trainer.fit(model, train_loader, val_loader)

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print(" No best model saved. Using last model state for evaluation.")
        model.to(device)
    else:
        print(f" Best model loaded from: {best_model_path}")
        model = MCDropoutNet.load_from_checkpoint(best_model_path).to(device)

    onnx_path = "./radar_models/radar_model.onnx"
    export_to_onnx(model, onnx_path)

    benchmark_results = comprehensive_benchmark(model, val_loader)
    onnx_avg_time, _ = benchmark_onnx_model(onnx_path, val_loader)
    noise_results = test_noise_robustness(model)

    print("\n FINAL SUMMARY:")
    print("=" * 60)
    print(f" Model trained with {n_tracks:,} samples.")
    print(f" Best Validation Accuracy: {benchmark_results['classification_accuracy']:.4f}")
    print(f" Final Regression RMSE: {benchmark_results['regression_rmse']:.2f} meters")
    print(f" PyTorch inference (batch={val_loader.batch_size}): {benchmark_results['pytorch_inference_ms_per_batch']:.2f} ms")
    print(f" MC-Dropout inference (batch={val_loader.batch_size}): {benchmark_results['mc_dropout_inference_ms_per_batch']:.2f} ms")
    print(f" ONNX inference (batch={val_loader.batch_size}): {onnx_avg_time:.2f} ms")
    print(" Noise Robustness (Accuracy vs. Noise Multiplier):")
    for level, res in noise_results.items():
        print(f"   - {level}x Noise: {res['acc']:.4f}")
    print(f" Best model checkpoint and ONNX model saved in './radar_models'")
    print("=" * 60)

    return model, benchmark_results
