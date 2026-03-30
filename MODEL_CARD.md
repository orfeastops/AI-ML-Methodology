# Model Card for AI-ML-Methodology

## Model Details

- **Model Name**: MCDropoutNet
- **Model Type**: Multi-task neural network (classification + regression) with Monte Carlo Dropout uncertainty estimation
- **Architecture**: CNN → BiLSTM → Multihead Attention → Dual heads
- **Input**: Radar track data (range, Doppler velocity, azimuth, elevation) over time
- **Output**: Missile type classification (3 classes) + trajectory prediction + uncertainty estimates
- **Training Data**: Simulated ballistic trajectories with noise

## Intended Use

- Ballistic missile detection and classification from radar data
- Trajectory prediction for early warning systems
- Uncertainty quantification for decision-making in critical scenarios

## Performance

- Classification Accuracy: ~0.95 (validation)
- Regression RMSE: ~50 meters (denormalized)
- Inference Time: ~10ms per batch (PyTorch), ~5ms (ONNX)
- Noise Robustness: Maintains >0.8 accuracy up to 2x noise levels

## Limitations

- Trained on simulated data only; real-world performance may vary
- Assumes specific radar parameters and noise models
- Not tested on real missile data

## Ethical Considerations

- Designed for defensive military applications
- Includes uncertainty estimation to avoid false positives in critical decisions
- Should not be used for offensive purposes

## Maintenance

- Version: 0.1.0
- Last Updated: 2026-03-30
- Contact: Repository maintainer