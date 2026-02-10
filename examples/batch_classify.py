#!/usr/bin/env python3
"""
Batch Classification Example
Scan multiple frequencies and classify each one
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classify_live import classify_signal

# Common frequencies to scan
frequencies = {
    'ADS-B': 1090e6,
    'NOAA 19': 137.62e6,
    'ISM Sensors': 433.92e6,
    'FM Radio': 98.7e6,
    'NOAA Weather': 162.4e6,
    'Pager': 152.84e6,
    'APRS': 144.39e6,
}

print("="*60)
print("RTL-ML BATCH CLASSIFICATION")
print("="*60)

results = []
for name, freq in frequencies.items():
    print(f"\n{name} ({freq/1e6:.2f} MHz)...")
    try:
        prediction, confidence, _ = classify_signal(freq, '../models/rtl_classifier_validated.pkl')
        results.append((name, prediction, confidence))
        print(f"  → {prediction} ({confidence*100:.0f}% confidence)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, prediction, confidence in results:
    match = "✓" if name.split()[0].upper() in prediction.upper() else "✗"
    print(f"{match} {name:15} → {prediction:15} ({confidence*100:.0f}%)")
