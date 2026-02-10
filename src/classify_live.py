#!/usr/bin/env python3
"""
RTL-ML Live Classification Script
Real-time signal identification using trained model
"""
from rtlsdr import RtlSdr
import numpy as np
import pickle
import time


class SignalFeatureExtractor:
    """Extract ML features from IQ signal data"""
    
    def extract_features(self, iq_samples):
        """Extract comprehensive feature set from IQ samples"""
        samples = np.array(iq_samples)
        i_signal = np.real(samples)
        q_signal = np.imag(samples)
        features = []
        power = np.abs(samples) ** 2
        features.extend([np.mean(power), np.std(power), np.max(power), np.min(power), np.median(power)])
        fft = np.fft.fft(samples)
        fft_mag = np.abs(fft)
        fft_power = fft_mag ** 2
        features.extend([np.mean(fft_power), np.std(fft_power), np.max(fft_power), np.sum(fft_power)])
        freqs = np.fft.fftfreq(len(samples))
        spectral_centroid = np.sum(freqs * fft_power) / np.sum(fft_power)
        features.append(spectral_centroid)
        features.extend([np.mean(i_signal), np.std(i_signal), np.mean(q_signal), np.std(q_signal)])
        phase = np.angle(samples)
        features.extend([np.mean(phase), np.std(phase)])
        inst_freq = np.diff(np.unwrap(phase))
        features.extend([np.mean(inst_freq), np.std(inst_freq)])
        return np.array(features)

def load_model():
    """Load trained classifier"""
    with open('rtl_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def classify_signal(sdr, model_data, frequency, duration=0.5):
    """Capture and classify a signal"""
    # Set frequency
    sdr.center_freq = frequency
    time.sleep(0.1)  # Let tuner settle
    
    # Capture samples
    num_samples = int(sdr.sample_rate * duration)
    samples = sdr.read_samples(num_samples)
    
    # Extract features
    features = model_data['feature_extractor'].extract_features(samples)
    features_scaled = model_data['scaler'].transform([features])
    
    # Classify
    prediction = model_data['model'].predict(features_scaled)[0]
    probabilities = None
    
    # Get probabilities if available
    if hasattr(model_data['model'], 'predict_proba'):
        probabilities = model_data['model'].predict_proba(features_scaled)[0]
    
    class_name = model_data['class_names'][prediction]
    
    return class_name, probabilities

def main():
    print("="*60)
    print("RTL-ML LIVE SIGNAL CLASSIFIER")
    print("="*60)
    
    # Load model
    print("\n📦 Loading trained model...")
    model_data = load_model()
    print(f"   Model: {type(model_data['model']).__name__}")
    print(f"   Classes: {', '.join(model_data['class_names'])}")
    
    # Initialize SDR
    print("\n📡 Initializing RTL-SDR...")
    sdr = RtlSdr()
    sdr.sample_rate = 1.024e6
    sdr.gain = 40
    print(f"   Sample rate: {sdr.sample_rate/1e6:.3f} MSPS")
    print(f"   Gain: {sdr.gain} dB")
    
    # Test frequencies
    test_freqs = [
        (98.7e6, "Rock FM"),
        (89.3e6, "NPR FM"),
        (162.4e6, "NOAA Weather"),
        (145.0e6, "Empty (noise)")
    ]
    
    print("\n" + "="*60)
    print("🔍 TESTING CLASSIFICATION")
    print("="*60)
    
    for freq, label in test_freqs:
        print(f"\n📻 {label} ({freq/1e6:.1f} MHz)")
        print(f"   Capturing...")
        
        class_name, probs = classify_signal(sdr, model_data, freq)
        
        print(f"   ✅ Predicted: {class_name}")
        
        if probs is not None:
            print(f"   Confidence:")
            for i, prob in enumerate(probs):
                class_label = model_data['class_names'][i]
                bar = "█" * int(prob * 20)
                print(f"      {class_label:15s}: {bar:20s} {prob*100:.1f}%")
        
        time.sleep(0.2)
    
    # Close SDR
    sdr.close()
    
    print("\n" + "="*60)
    print("✅ CLASSIFICATION COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()
