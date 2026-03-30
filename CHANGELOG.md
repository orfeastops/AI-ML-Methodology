# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-30

### Added
- Initial release of AI-ML Methodology repository
- Ballistic trajectory simulation with physics-based model
- MCDropoutNet: CNN+LSTM+Attention architecture for classification and regression
- Monte Carlo Dropout for uncertainty estimation
- ONNX export and benchmarking utilities
- Noise robustness testing
- PyTorch Lightning training pipeline
- Modular package structure (`ai_ml_methodology/`)
- CLI interface with argparse
- Weights & Biases integration
- Demo mode for quick testing
- GitHub Actions CI workflow
- Unit tests
- Model card, contributing guidelines, code of conduct

### Changed
- Refactored monolithic script into modular package
- Added professional documentation and metadata

### Technical Details
- Dataset: 50,000 simulated tracks (demo: 1,000)
- Model: 256 hidden dim, 0.25 dropout, 25 MC samples
- Training: AdamW optimizer, ReduceLROnPlateau scheduler
- Validation: Early stopping, model checkpointing