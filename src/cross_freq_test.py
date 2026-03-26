#!/usr/bin/env python3
"""Cross-frequency validation: Does the model recognize FM on a new frequency?
If yes -> learned FM characteristics. If no -> memorized 98.7 MHz."""
import numpy as np
import pickle
import os
from rtlsdr import RtlSdr

# Feature extractor (must match training)
def extract_features(samples):
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
    features.append(np.argmax(fft_power) / len(fft_power))
    
    i_s = np.real(samples)
    q_s = np.imag(samples)
    features.append(np.mean(i_s))
    features.append(np.std(i_s))
    features.append(np.mean(q_s))
    features.append(np.std(q_s))
    
    phase = np.angle(samples)
    features.append(np.mean(phase))
    features.append(np.std(phase))
    
    phase_diff = np.diff(phase)
    features.append(np.mean(phase_diff))
    features.append(np.std(phase_diff))
    
    bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
    features.append(bandwidth / len(fft_power))
    
    return np.array(features)

# Load model
model_path = os.path.expanduser('~/rtl-ml/rtl_classifier_validated.pkl')
with open(model_path, 'rb') as f:
    pkg = pickle.load(f)
model = pkg['model']
scaler = pkg['scaler']
print(f"Loaded model: {pkg['model_name']}")

# Test frequencies - different FM stations
test_freqs = [
    (101.1e6, "FM 101.1 MHz (new station)"),
    (93.3e6,  "FM 93.3 MHz (new station)"),
    (88.5e6,  "FM 88.5 MHz (new station)"),
    (98.7e6,  "FM 98.7 MHz (training freq - control)"),
    (145.0e6, "145.0 MHz (noise baseline - control)"),
]

sdr = RtlSdr()
sdr.sample_rate = 1.024e6
sdr.gain = 'auto'

print("\n" + "=" * 70)
print("CROSS-FREQUENCY FM VALIDATION")
print("=" * 70)
print("Model trained on FM @ 98.7 MHz. Testing other FM stations...\n")

results = []
for freq, desc in test_freqs:
    sdr.center_freq = freq
    
    # Capture 10 samples and classify each
    predictions = []
    for i in range(10):
        samples = sdr.read_samples(512 * 1024)
        samples = samples - np.mean(samples)  # DC removal
        feats = extract_features(samples).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        pred = model.predict(feats_scaled)[0]
        predictions.append(pred)
    
    # Count predictions
    from collections import Counter
    counts = Counter(predictions)
    majority = counts.most_common(1)[0]
    
    is_fm = "FM" in desc
    classified_fm = majority[0] == "FM_broadcast"
    status = ""
    if is_fm and classified_fm:
        status = "PASS - correctly identified as FM"
    elif is_fm and not classified_fm:
        status = "FAIL - FM not recognized (memorized frequency?)"
    elif not is_fm and not classified_fm:
        status = "PASS - correctly NOT classified as FM"
    elif not is_fm and classified_fm:
        status = "FAIL - non-FM classified as FM"
    
    print(f"  {desc}")
    print(f"    Predictions: {dict(counts)}")
    print(f"    Majority: {majority[0]} ({majority[1]}/10)")
    print(f"    {status}")
    print()
    results.append((desc, classified_fm if is_fm else not classified_fm))

sdr.close()

# Summary
passed = sum(1 for _, ok in results if ok)
total = len(results)
print("=" * 70)
print(f"RESULT: {passed}/{total} tests passed")
if passed == total:
    print("MODEL LEARNED FM CHARACTERISTICS (not frequency memorization)")
elif passed >= 3:
    print("MODEL MOSTLY GENERALIZED (some frequency sensitivity)")
else:
    print("MODEL MEMORIZED FREQUENCY (v1 problem persists)")
