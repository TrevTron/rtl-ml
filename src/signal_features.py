#!/usr/bin/env python3
"""
Signal Feature Extractor
Extracts 18 numerical features from IQ samples for ML classification
"""
import numpy as np

class SignalFeatureExtractor:
    """Extract 18 features from raw IQ samples."""
    
    def __init__(self, sample_rate=1.024e6):
        self.sample_rate = sample_rate
    
    def extract_features(self, samples):
        """
        Extract 18 features from IQ samples.
        
        Args:
            samples: Complex IQ samples (numpy array)
            
        Returns:
            numpy array of 18 numerical features
        """
        features = []
        
        # Power calculations
        power = np.abs(samples) ** 2
        features.append(np.mean(power))
        features.append(np.std(power))
        features.append(np.max(power))
        features.append(np.min(power))
        
        # FFT analysis
        fft_vals = np.fft.fft(samples)
        fft_power = np.abs(fft_vals) ** 2
        features.append(np.mean(fft_power))
        features.append(np.std(fft_power))
        features.append(np.max(fft_power))
        
        # Peak frequency index (normalized)
        peak_freq_idx = np.argmax(fft_power)
        features.append(peak_freq_idx / len(fft_power))
        
        # I/Q component statistics
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        features.append(np.mean(i_samples))
        features.append(np.std(i_samples))
        features.append(np.mean(q_samples))
        features.append(np.std(q_samples))
        
        # Phase analysis
        phase = np.angle(samples)
        features.append(np.mean(phase))
        features.append(np.std(phase))
        
        # Instantaneous frequency (phase derivative)
        phase_diff = np.diff(phase)
        features.append(np.mean(phase_diff))
        features.append(np.std(phase_diff))
        
        # Bandwidth estimation
        bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
        features.append(bandwidth / len(fft_power))
        
        return np.array(features)
    
    @property
    def feature_names(self):
        """List of feature names for documentation."""
        return [
            'power_mean', 'power_std', 'power_max', 'power_min',
            'fft_mean', 'fft_std', 'fft_max', 'fft_peak_idx',
            'i_mean', 'i_std', 'q_mean', 'q_std',
            'phase_mean', 'phase_std', 'phase_diff_mean', 'phase_diff_std',
            'bandwidth_ratio'
        ]
