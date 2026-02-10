#!/usr/bin/env python3
"""
Quick Start Example
Minimal code to classify a radio signal using pre-trained model
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classify_live import classify_signal

# Example: Classify FM broadcast at 98.7 MHz
print("Classifying FM radio at 98.7 MHz...")
prediction, confidence, _ = classify_signal(9870000, '../models/rtl_classifier_validated.pkl')

print(f"\nDetected: {prediction}")
print(f"Confidence: {confidence*100:.0f}%")
