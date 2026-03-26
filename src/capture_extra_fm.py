#!/usr/bin/env python3
"""Capture additional FM samples from multiple frequencies for generalization."""
import numpy as np
import os
import time
from rtlsdr import RtlSdr
from datetime import datetime
from scipy import signal as sig

base = os.path.expanduser('~/rtl-ml/datasets_validated/FM_broadcast')

# Additional FM stations to capture (different parts of the band)
extra_freqs = [
    (88.5e6,  "FM_low_band"),
    (93.3e6,  "FM_mid_low"),
    (101.1e6, "FM_mid_high"),
    (105.7e6, "FM_high_band"),
]

SAMPLES_PER_FREQ = 25  # 25 each x 4 freqs = 100 additional samples
SAMPLE_SIZE = 512 * 1024
SAMPLE_RATE = 1.024e6
SPACING = 2.0  # seconds between captures

def validate_fm(samples, sr):
    """Check if this looks like FM broadcast."""
    s = samples - np.mean(samples)
    fft = np.abs(np.fft.fft(s))
    bw_bins = np.sum(fft > fft.mean() * 3)
    bw_khz = bw_bins * sr / len(fft) / 1000
    
    noise_floor = np.median(np.abs(s) ** 2)
    signal_power = np.mean(np.abs(s) ** 2)
    snr_db = 10 * np.log10(signal_power / max(noise_floor, 1e-30))
    
    return bw_khz > 20, snr_db, bw_khz  # Relaxed from 50 to 20 for weaker stations

sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.gain = 'auto'

total_saved = 0
total_rejected = 0

for freq, desc in extra_freqs:
    sdr.center_freq = freq
    freq_mhz = freq / 1e6
    
    # Quick signal check
    test = sdr.read_samples(SAMPLE_SIZE)
    test = test - np.mean(test)
    i_std = test.real.std()
    print(f"\n{'='*60}")
    print(f"  {desc} @ {freq_mhz:.1f} MHz  (I std: {i_std:.4f})")
    
    if i_std < 0.05:
        print(f"  SKIPPING - too weak (I std {i_std:.4f} < 0.05)")
        continue
    
    saved = 0
    rejected = 0
    for i in range(SAMPLES_PER_FREQ * 3):  # up to 3x attempts
        if saved >= SAMPLES_PER_FREQ:
            break
        
        samples = sdr.read_samples(SAMPLE_SIZE)
        samples = samples - np.mean(samples)
        
        valid, snr_db, bw_khz = validate_fm(samples, SAMPLE_RATE)
        
        if valid:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            fname = f'FM_broadcast_{ts}.npy'
            data = {
                'samples': samples,
                'center_freq': freq,
                'sample_rate': SAMPLE_RATE,
                'timestamp': ts,
                'label': 'FM_broadcast',
                'duration': 0.5,
                'snr_db': snr_db,
                'version': 'v2',
            }
            np.save(os.path.join(base, fname), data)
            saved += 1
            total_saved += 1
        else:
            rejected += 1
            total_rejected += 1
        
        time.sleep(SPACING)
    
    print(f"  Saved: {saved}/{SAMPLES_PER_FREQ}  Rejected: {rejected}")

sdr.close()

print(f"\n{'='*60}")
print(f"TOTAL: {total_saved} new FM samples added, {total_rejected} rejected")
print(f"FM_broadcast now has {len(os.listdir(base))} total samples")
