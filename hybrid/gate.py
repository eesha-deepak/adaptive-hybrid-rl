import numpy as np
from typing import Dict, List, Tuple

class SwitchingGate:
    """
    Confidence-based gate for switching between model-based and model-free
    """
    
    def __init__(self, 
                 threshold: float = 0.1,
                 calibration_quantile: float = 0.9,
                 buffer_size: int = 100,
                 hysteresis_factor: float = 0.2):
        self.threshold = threshold
        self.calibration_quantile = calibration_quantile
        self.buffer_size = buffer_size
        self.hysteresis_factor = hysteresis_factor

        self.disagreements = []
        self.prediction_errors = []

        self.last_mode = "model-free"
        self.consecutive_mb = 0
        self.consecutive_mf = 0
        
    def should_use_mb(self, disagreement: float) -> bool:
        """
        Decide whether to use model-based control
        """
        if self.last_mode == "model-based":
            use_mb = disagreement < self.threshold * (1 + self.hysteresis_factor)
        else:
            use_mb = disagreement < self.threshold * (1 - self.hysteresis_factor)
        
        if use_mb:
            self.consecutive_mb += 1
            self.consecutive_mf = 0
            self.last_mode = "model-based"
        else:
            self.consecutive_mf += 1
            self.consecutive_mb = 0
            self.last_mode = "model-free"
        
        return use_mb
    
    def add_calibration_data(self, disagreement: float, actual_error: float):
        self.disagreements.append(disagreement)
        self.prediction_errors.append(actual_error)

        if len(self.disagreements) > self.buffer_size:
            self.disagreements.pop(0)
            self.prediction_errors.pop(0)
    
    def calibrate_threshold(self) -> Dict:
        """
        Calibrate threshold based on empirical coverage
        """
        if len(self.disagreements) < 50:
            return {
                "calibrated": False,
                "reason": "insufficient_data",
                "n_samples": len(self.disagreements)
            }
        sorted_pairs = sorted(zip(self.disagreements, self.prediction_errors))
        disagreements_sorted = [d for d, _ in sorted_pairs]
        errors_sorted = [e for _, e in sorted_pairs]
        
        n = len(errors_sorted)
        idx = int(self.calibration_quantile * n)
        
        median_error = np.median(errors_sorted[:idx])
        
        old_threshold = self.threshold
        self.threshold = disagreements_sorted[idx]
        
        return {
            "calibrated": True,
            "old_threshold": old_threshold,
            "new_threshold": self.threshold,
            "target_quantile": self.calibration_quantile,
            "median_error": median_error,
            "n_samples": n
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about gate behavior"""
        return {
            "threshold": self.threshold,
            "last_mode": self.last_mode,
            "consecutive_mb": self.consecutive_mb,
            "consecutive_mf": self.consecutive_mf,
            "n_calibration_samples": len(self.disagreements),
            "mean_disagreement": np.mean(self.disagreements) if self.disagreements else 0,
            "mean_error": np.mean(self.prediction_errors) if self.prediction_errors else 0
        }
    
    def get_coverage_stats(self) -> Dict:
        """
        Compute empirical coverage statistics
        
        Returns:
            Coverage at different disagreement levels
        """
        if len(self.disagreements) < 10:
            return {}
        
        below_threshold = [
            (d, e) for d, e in zip(self.disagreements, self.prediction_errors)
            if d < self.threshold
        ]
        
        if not below_threshold:
            return {"coverage": 0.0, "n_below_threshold": 0}
        
        errors_below = [e for _, e in below_threshold]
        median_error = np.median(errors_below)

        accurate = sum(1 for e in errors_below if e < median_error)
        coverage = accurate / len(errors_below)
        
        return {
            "coverage": coverage,
            "n_below_threshold": len(below_threshold),
            "n_total": len(self.disagreements),
            "median_error_at_threshold": median_error
        }