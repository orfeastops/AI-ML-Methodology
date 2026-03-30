import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from tqdm.auto import tqdm
from ai_ml_methodology.data import NOISE_STD, MISSILE_SPECS, simulate_track, DATA_DIR
from ai_ml_methodology.model import MCDropoutNet

try:
    import onnxruntime_gpu
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("ONNX Runtime will use CUDAExecutionProvider.")
except ImportError:
    providers = ['CPUExecutionProvider']
    print("ONNX Runtime will use CPUExecutionProvider.")

# ONNX Export & Optimization
def export_to_onnx(model, save_path="radar_model.onnx"):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 600, 4, device=device)

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
        norm_stats = __import__('pickle').load(f)

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
        with open(f"./radar_models/benchmark_results.pkl", 'wb') as f: __import__('pickle').dump(results, f)

    return results

# Noise Robustness Testing
def test_noise_robustness(model, base_noise_std=NOISE_STD, n_samples=500):
    print("🔬 Testing noise robustness...")
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = {}

    device = next(model.parameters()).device
    model.eval()

    with open(f"{DATA_DIR}/norm_stats.pkl", "rb") as f: norm = __import__('pickle').load(f)
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