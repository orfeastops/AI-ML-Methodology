## Dual-Mode Radar Data System

This project supports multiple radar data sources with a unified ML pipeline. Switch between them easily:

### Supported Data Sources

| Source | Type | Classes | Use Case |
|--------|------|---------|----------|
| **missile** | Simulated | ATACMS, Iskander, GMLRS | Research, methodology development |
| **aircraft** | Real-ready | Boeing 737, A320, Cessna 172 | Real-world validation, production scenarios |

### Usage Examples

#### Train on Missile Data (Simulated)
```bash
python main.py --mode train --data-source missile --demo
```
Output: Trained on 1,000 simulated ballistic trajectories

#### Train on Aircraft Data (Real-like)
```bash
python main.py --mode train --data-source aircraft --demo
```
Output: Trained on 1,000 aircraft-like trajectories (currently synthetic, ready for OpenSky integration)

#### Full Production Run with W&B Tracking
```bash
python main.py --mode train --data-source aircraft --wandb
```
Logs: Real aircraft model on 50K samples to Weights & Biases

### Adding New Data Sources

To add a new radar data source (e.g., real OpenSky, defense datasets):

```python
# ai_ml_methodology/data_sources.py

from ai_ml_methodology.data_sources import RadarDataSource

class MyCustomRadarSource(RadarDataSource):
    def __init__(self, **kwargs):
        self.class_names = ["Target_Type_1", "Target_Type_2", ...]
    
    def load_data(self):
        # Load your data and convert to:
        # data: (N, T, 4) array [range, velocity, azimuth, elevation]
        # labels: (N,) class indices
        # targets: (N, T) future position predictions
        return data, labels, targets
    
    def get_class_names(self):
        return self.class_names

# Register in get_data_source() factory:
sources = {
    "missile": BalisticMissileSource,
    "aircraft": OpenSkyAircraftSource,
    "my_source": MyCustomRadarSource,  # Add here
}
```

Then use it:
```bash
python main.py --mode train --data-source my_source
```

### OpenSky Aircraft Integration (TODO)

The `OpenSkyAircraftSource` is currently using synthetic aircraft-like data. To connect real data:

1. Get OpenSky API credentials: https://opensky-network.org/api/index.html
2. Uncomment implementation in `ai_ml_methodology/data_sources.py`
3. Set environment variable: `export OPENSKY_USERNAME=your_username`
4. Run: `python main.py --mode train --data-source aircraft`

### Data Format

All sources must output data in this unified format:

```
data: np.ndarray shape (N, T, 4)
    Channel 0: Range (meters)
    Channel 1: Radial Velocity (m/s, Doppler)
    Channel 2: Azimuth angle (radians)
    Channel 3: Elevation angle (radians)
    T = 600 timesteps @ 0.1s = 60 seconds

labels: np.ndarray shape (N,)
    Integer class IDs (0, 1, 2, ...)

targets: np.ndarray shape (N, T)
    Future range position at each timestep (for regression)
```

### Comparing Models Across Data Sources

```bash
# Train on missiles
python main.py --mode train --data-source missile --wandb

# Train on aircraft
python main.py --mode train --data-source aircraft --wandb

# View comparison in W&B dashboard
# Compare accuracy, uncertainty, inference time across domains
```

### Validation & Testing

Both data sources use the same model architecture and evaluation metrics:
- Classification accuracy
- Regression RMSE (trajectory prediction)
- MC-Dropout uncertainty calibration
- Robustness to noise

### Future Enhancements

- [ ] Full OpenSky API integration
- [ ] Real radar data sources (research partnerships)
- [ ] Domain adaptation techniques (missile ↔ aircraft)
- [ ] Multi-source training (combined datasets)
- [ ] Synthetic-to-real transfer learning
