#!/usr/bin/env python3
"""
RTL-ML Live Classification Script
Real-time signal identification using trained v2 model
"""
from rtlsdr import RtlSdr
import numpy as np
import pickle
import time
import argparse
import os


def extract_features(samples):
    """Extract 17 features from IQ samples (must match train_validated.py)."""
    features = []
    power = np.abs(samples) ** 2
    features.append(np.mean(power))
    features.append(np.std(power))
    features.append(np.max(power))
    features.append(np.min(power))

    fft_vals = np.fft.fft(samples)
    fft_power = np.abs(fft_vals) ** 2
    features.append(np.mean(fft_power))
    features.append(np.std(fft_power))
    features.append(np.max(fft_power))

    peak_freq_idx = np.argmax(fft_power)
    features.append(peak_freq_idx / len(fft_power))

    i_samples = np.real(samples)
    q_samples = np.imag(samples)
    features.append(np.mean(i_samples))
    features.append(np.std(i_samples))
    features.append(np.mean(q_samples))
    features.append(np.std(q_samples))

    phase = np.angle(samples)
    features.append(np.mean(phase))
    features.append(np.std(phase))

    phase_diff = np.diff(phase)
    features.append(np.mean(phase_diff))
    features.append(np.std(phase_diff))

    bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
    features.append(bandwidth / len(fft_power))

    return np.array(features)


def load_model(model_path=None):
    """Load trained v2 classifier."""
    if model_path is None:
        # Search common locations
        candidates = [
            'models/rtl_classifier_validated.pkl',
            'rtl_classifier_validated.pkl',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'rtl_classifier_validated.pkl'),
        ]
        for path in candidates:
            if os.path.exists(path):
                model_path = path
                break
        else:
            raise FileNotFoundError(
                "Model not found. Expected at models/rtl_classifier_validated.pkl\n"
                "Train one with: python src/train_validated.py"
            )

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def classify_signal(frequency, model_path=None, duration=0.5, sdr=None):
    """Capture and classify a signal at the given frequency.

    Args:
        frequency: Center frequency in Hz (e.g. 98.7e6)
        model_path: Path to .pkl model file (auto-detected if None)
        duration: Capture duration in seconds
        sdr: Existing RtlSdr instance (opens new one if None)

    Returns:
        (class_name, confidence, probabilities)
    """
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = list(model.classes_)

    close_sdr = False
    if sdr is None:
        sdr = RtlSdr()
        sdr.sample_rate = 1.024e6
        sdr.gain = 'auto'
        close_sdr = True

    sdr.center_freq = frequency
    time.sleep(0.3)
    _ = sdr.read_samples(1024)  # flush

    num_samples = int(sdr.sample_rate * duration)
    samples = sdr.read_samples(num_samples)
    samples = samples - np.mean(samples)  # DC removal

    features = extract_features(samples)
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    confidence = 0.0
    probabilities = None

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(probabilities))

    if close_sdr:
        sdr.close()

    return prediction, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description='RTL-ML Live Signal Classifier')
    parser.add_argument('--freq', type=float, help='Single frequency to classify (Hz, e.g. 98.7e6)')
    parser.add_argument('--model', type=str, default=None, help='Path to model .pkl file')
    args = parser.parse_args()

    print("=" * 60)
    print("RTL-ML LIVE SIGNAL CLASSIFIER (v2)")
    print("=" * 60)

    model_data = load_model(args.model)
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = list(model.classes_)

    print(f"\n📦 Model: {model_data['model_name']}")
    print(f"   Classes: {', '.join(class_names)}")

    print("\n📡 Initializing RTL-SDR...")
    sdr = RtlSdr()
    sdr.sample_rate = 1.024e6
    sdr.gain = 'auto'
    print(f"   Sample rate: {sdr.sample_rate/1e6:.3f} MSPS")
    print(f"   Gain: auto")

    if args.freq:
        test_freqs = [(args.freq, f"User freq ({args.freq/1e6:.1f} MHz)")]
    else:
        test_freqs = [
            (98.7e6, "FM Broadcast"),
            (162.4e6, "NOAA Weather"),
            (144.39e6, "APRS"),
            (433.92e6, "ISM Sensors"),
            (152.84e6, "Pager"),
            (462.5625e6, "FRS/GMRS"),
            (145.0e6, "Noise Baseline"),
        ]

    print("\n" + "=" * 60)
    print("🔍 CLASSIFYING SIGNALS")
    print("=" * 60)

    for freq, label in test_freqs:
        print(f"\n📻 {label} ({freq/1e6:.1f} MHz)")
        print(f"   Capturing...")

        sdr.center_freq = freq
        time.sleep(0.3)
        _ = sdr.read_samples(1024)

        samples = sdr.read_samples(int(sdr.sample_rate * 0.5))
        samples = samples - np.mean(samples)

        features = extract_features(samples)
        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]

        print(f"   ✅ Predicted: {prediction}")

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            print(f"   Confidence:")
            for i, prob in enumerate(probs):
                bar = "█" * int(prob * 20)
                print(f"      {class_names[i]:15s}: {bar:20s} {prob*100:.1f}%")

        time.sleep(0.2)

    sdr.close()

    print("\n" + "=" * 60)
    print("✅ CLASSIFICATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
