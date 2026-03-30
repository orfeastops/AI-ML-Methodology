"""
Data source abstraction for flexible radar data handling.
Supports both simulated ballistic trajectories and real aircraft tracking.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List


class RadarDataSource(ABC):
    """Abstract base for all radar data sources."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load radar data.
        Returns: (data, labels, targets)
            data: (N, T, 4) array [range, radial_velocity, azimuth, elevation]
            labels: (N,) class labels
            targets: (N, T) future range predictions
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        pass


class BalisticMissileSource(RadarDataSource):
    """Existing simulated missile data."""
    
    def __init__(self, n_samples=50000):
        from ai_ml_methodology.data import generate_all, ensure_data_exists, N_TRACKS, MISSILE_SPECS
        self.n_samples = n_samples
        self.missile_specs = MISSILE_SPECS
        ensure_data_exists(n_samples)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import os
        import numpy as np
        
        data_dir = "./radar_data"
        all_data, all_labels, all_targets = [], [], []
        
        data_files = sorted([f for f in os.listdir(data_dir) 
                            if f.startswith("data_") and f.endswith(".npy")])
        
        for file in data_files:
            file_num = int(file.split('_')[1].split('.')[0])
            data = np.load(f"{data_dir}/data_{file_num:03d}.npy")
            labels = np.load(f"{data_dir}/label_{file_num:03d}.npy")
            targets = np.load(f"{data_dir}/target_{file_num:03d}.npy")
            
            all_data.append(data)
            all_labels.append(labels)
            all_targets.append(targets)
        
        return (np.concatenate(all_data), 
                np.concatenate(all_labels),
                np.concatenate(all_targets))
    
    def get_class_names(self) -> List[str]:
        return [spec["name"] for spec in self.missile_specs.values()]


class OpenSkyAircraftSource(RadarDataSource):
    """Real aircraft radar data from OpenSky Network."""
    
    def __init__(self, n_samples=50000, cache_dir="./opensky_cache"):
        self.n_samples = n_samples
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        # Aircraft types to classify
        self.aircraft_types = {
            0: "Boeing 737",
            1: "Airbus A320", 
            2: "Cessna 172",
        }
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads historical OpenSky data and converts to radar format.
        This is a placeholder; actual implementation would:
        1. Query OpenSky API or download historical data
        2. Extract position/velocity over time windows
        3. Convert to radar measurement space (range, Doppler, angles)
        """
        try:
            import requests
            import json
        except ImportError:
            raise ImportError("Install: pip install requests")
        
        print("🌐 Fetching OpenSky aircraft data...")
        print("   Note: Full implementation requires OpenSky API key")
        print("   Visit: https://opensky-network.org/api/index.html")
        
        # TODO: Implement actual OpenSky data fetching
        # For now, return synthetic data that mimics aircraft behavior
        return self._generate_synthetic_aircraft_data()
    
    def _generate_synthetic_aircraft_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate aircraft-like trajectories for demonstration."""
        from ai_ml_methodology.data import simulate_track
        
        data, labels, targets = [], [], []
        
        for i in range(self.n_samples):
            # Simulate aircraft (less aggressive trajectory than missiles)
            aircraft_type = i % 3
            
            # Aircraft specs: slower, higher altitude
            v0 = 250 + np.random.normal(0, 20)  # ~250 m/s cruise
            theta = 5 + np.random.normal(0, 2)  # Shallow climb/descent
            
            track = simulate_track(v0, theta)
            data.append(track)
            labels.append(aircraft_type)
            
            # Future position target
            idx = np.arange(600) + 5
            idx[idx >= 600] = 599
            targets.append(track[idx, 0])
        
        return np.array(data), np.array(labels), np.array(targets)
    
    def get_class_names(self) -> List[str]:
        return list(self.aircraft_types.values())


def get_data_source(source_type: str = "missile", **kwargs) -> RadarDataSource:
    """Factory function to get radar data source."""
    sources = {
        "missile": BalisticMissileSource,
        "aircraft": OpenSkyAircraftSource,
    }
    
    if source_type not in sources:
        raise ValueError(f"Unknown source: {source_type}. Choose from {list(sources.keys())}")
    
    return sources[source_type](**kwargs)
