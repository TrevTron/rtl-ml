#!/usr/bin/env python3
"""SDR diagnostic - test actual signal reception quality"""
from rtlsdr import RtlSdr
import numpy as np
import time

sdr = RtlSdr()
sdr.sample_rate = 1.024e6

test_freqs = [
    (98.7e6,   'FM_broadcast'),
    (162.4e6,  'NOAA_weather'),
    (144.39e6, 'APRS'),
    (433.92e6, 'ISM_sensors'),
    (152.84e6, 'pager'),
    (462.5625e6, 'FRS_GMRS'),
    (145.0e6,  'noise_baseline'),
]

for gain_mode in [40, 'auto']:
    print(f"\n{'='*60}")
    print(f"GAIN MODE: {gain_mode}")
    print(f"{'='*60}")
    sdr.gain = gain_mode

    for freq, label in test_freqs:
        sdr.center_freq = freq
        time.sleep(0.5)
        _ = sdr.read_samples(2048)  # flush
        time.sleep(0.3)
        samples = sdr.read_samples(int(1.024e6 * 0.5))

        # Raw analysis (before DC removal)
        fft_raw = np.abs(np.fft.fft(samples)) ** 2
        dc_power = fft_raw[0]
        raw_max = np.max(fft_raw)
        raw_95 = np.percentile(fft_raw, 95)
        raw_median = np.median(fft_raw)

        # DC-removed analysis
        samples_clean = samples - np.mean(samples)
        fft_clean = np.abs(np.fft.fft(samples_clean)) ** 2
        clean_max = np.max(fft_clean)
        clean_95 = np.percentile(fft_clean, 95)
        clean_median = np.median(fft_clean)
        snr_clean = 10 * np.log10(clean_95 / max(clean_median, 1e-20))

        # Bandwidth
        bw_bins = np.sum(fft_clean > clean_max * 0.1)
        bw_khz = bw_bins / len(fft_clean) * 1024

        # I/Q stats
        i_mean = np.mean(np.real(samples))
        q_mean = np.mean(np.imag(samples))
        i_std = np.std(np.real(samples))

        print(f"\n  {label:15s} @ {freq/1e6:.2f} MHz")
        print(f"    DC bin power:    {dc_power:.2e}")
        print(f"    Raw max power:   {raw_max:.2e}")
        print(f"    DC dominance:    {dc_power/max(raw_95,1e-20):.1f}x over 95th pctile")
        print(f"    Clean SNR:       {snr_clean:.1f} dB")
        print(f"    Clean BW:        {bw_khz:.0f} kHz")
        print(f"    I mean/std:      {i_mean:.6f} / {i_std:.6f}")
        print(f"    Q mean:          {q_mean:.6f}")

        # Verdict
        if label == 'FM_broadcast' and bw_khz < 50:
            print(f"    VERDICT: BAD - FM should be >50 kHz BW")
        elif label == 'FM_broadcast' and bw_khz > 50:
            print(f"    VERDICT: GOOD - real FM signal detected")
        elif label == 'noise_baseline':
            print(f"    VERDICT: baseline reference")
        elif snr_clean < 6:
            print(f"    VERDICT: WEAK/NO SIGNAL")
        else:
            print(f"    VERDICT: signal present (SNR > 6 dB)")

sdr.close()
print("\nDiagnostic complete.")
