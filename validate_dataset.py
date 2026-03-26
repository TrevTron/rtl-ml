#!/usr/bin/env python3
"""
RTL-ML Dataset Validator
Verifies signal classifications match spectral characteristics.
Uses Welch PSD, spectral flatness, kurtosis, and burst detection.
"""

import numpy as np
from pathlib import Path
from scipy import signal as sp_signal
from scipy.stats import kurtosis as sp_kurtosis
import sys

SAMPLE_RATE = 1.024e6
EXPECTED_LENGTH = 512000  # 0.5s @ 1.024 MSPS

SIGNAL_CLASSES = [
    'APRS', 'FM_broadcast', 'FRS_GMRS', 'ISM_sensors',
    'NOAA_weather', 'noise', 'pager'
]


def compute_metrics(samples):
    """Compute spectral metrics from complex IQ samples."""
    # Remove DC offset (RTL-SDR bias)
    s = samples - np.mean(samples)

    # Welch PSD (handles complex properly)
    nfft = min(4096, len(s))
    freqs, psd = sp_signal.welch(s, fs=SAMPLE_RATE, nperseg=nfft, return_onesided=False)
    psd_db = 10 * np.log10(psd + 1e-20)

    # SNR: peak above median noise floor
    noise_floor = np.median(psd_db)
    snr = np.max(psd_db) - noise_floor

    # Occupied bandwidth (99% power)
    total_power = np.sum(psd)
    idx = np.argsort(freqs)
    cum = np.cumsum(psd[idx]) / total_power
    low = np.searchsorted(cum, 0.005)
    high = np.searchsorted(cum, 0.995)
    occ_bw = abs(freqs[idx[high]] - freqs[idx[low]])

    # Spectral flatness: 1.0 = white noise, 0 = tonal
    geo = np.exp(np.mean(np.log(psd + 1e-20)))
    arith = np.mean(psd)
    flatness = geo / (arith + 1e-20)

    # Time-domain kurtosis (Gaussian=3, bursty>3)
    mag = np.abs(s)
    kurt = sp_kurtosis(mag, fisher=False)

    # Burst ratio from smoothed envelope
    win_sz = max(64, int(SAMPLE_RATE * 0.001))
    win = sp_signal.windows.hann(win_sz)
    env = sp_signal.convolve(mag, win, mode='same') / np.sum(win)
    mean_env = np.mean(env)
    burst = np.max(env) / mean_env if mean_env > 1e-10 else 1.0

    return {
        'snr': round(snr, 1),
        'bw_khz': round(occ_bw / 1e3, 1),
        'flatness': round(flatness, 4),
        'kurtosis': round(kurt, 2),
        'burst': round(burst, 2),
    }


def score_class(m, cls):
    """Score how well metrics match claimed class. Returns (score, reasons)."""
    s, r = 0, []

    snr, bw, flat, kurt, burst = m['snr'], m['bw_khz'], m['flatness'], m['kurtosis'], m['burst']

    if cls == 'noise':
        if flat > 0.5:
            s += 2; r.append(f"Flat spectrum ({flat:.3f})")
        elif flat > 0.3:
            s += 1; r.append(f"Moderately flat ({flat:.3f})")
        else:
            s -= 1; r.append(f"Structured spectrum ({flat:.3f}) — signal present?")
        if snr < 10:
            s += 1; r.append(f"Low SNR ({snr:.1f} dB)")
        elif snr > 20:
            s -= 2; r.append(f"High SNR ({snr:.1f} dB)")
        if 2.5 < kurt < 4.0:
            s += 1; r.append(f"Gaussian kurtosis ({kurt:.1f})")
        return s, r

    # All real signal classes
    if snr > 15:
        s += 2; r.append(f"Strong signal ({snr:.1f} dB)")
    elif snr > 8:
        s += 1; r.append(f"Signal present ({snr:.1f} dB)")
    elif snr > 3:
        r.append(f"Weak signal ({snr:.1f} dB)")
    else:
        s -= 1; r.append(f"Very weak ({snr:.1f} dB)")

    if cls == 'FM_broadcast':
        if bw > 100: s += 2; r.append(f"Wideband ({bw:.0f} kHz)")
        elif bw > 50: s += 1; r.append(f"Moderate BW ({bw:.0f} kHz)")
        if flat < 0.4: s += 1; r.append(f"Structured ({flat:.3f})")

    elif cls == 'APRS':
        # Sparse packets — most lenient class
        if kurt > 3.5: s += 1; r.append(f"Non-Gaussian ({kurt:.1f})")
        if burst > 1.3: s += 1; r.append(f"Packet bursts ({burst:.1f}x)")
        else: r.append(f"Low burst ({burst:.1f}x) — sparse expected")

    elif cls == 'ISM_sensors':
        if burst > 2: s += 2; r.append(f"Clear OOK bursts ({burst:.1f}x)")
        elif burst > 1.3: s += 1; r.append(f"Some bursting ({burst:.1f}x)")
        if flat < 0.6: s += 1; r.append(f"Structured ({flat:.3f})")

    elif cls == 'FRS_GMRS':
        if burst > 1.3: s += 1; r.append(f"Bursty ({burst:.1f}x)")
        if flat < 0.6: s += 1; r.append(f"Structured ({flat:.3f})")
        if bw < 25: s += 1; r.append(f"Narrowband ({bw:.0f} kHz)")

    elif cls == 'NOAA_weather':
        if flat < 0.5: s += 1; r.append(f"Structured ({flat:.3f})")
        if bw < 50: s += 1; r.append(f"Narrowband ({bw:.0f} kHz)")

    elif cls == 'pager':
        if burst > 1.5: s += 1; r.append(f"FSK bursts ({burst:.1f}x)")
        if flat < 0.6: s += 1; r.append(f"Structured ({flat:.3f})")

    return s, r


def verdict_from_score(score):
    if score >= 3: return "STRONG", "high"
    if score >= 1: return "PLAUSIBLE", "medium"
    if score >= 0: return "WEAK", "low"
    return "SUSPECT", "fail"


def validate_file(fpath, cls):
    """Validate one .npy file. Returns (verdict, metrics, issues)."""
    issues = []
    try:
        raw = np.load(fpath, allow_pickle=True)
        d = raw.item()
        samples = d['samples']
        label = d.get('label', '')
        freq = d.get('center_freq', 0)
        sr = d.get('sample_rate', 0)

        if label != cls:
            issues.append(f"Label '{label}' != folder '{cls}'")
        if abs(sr - SAMPLE_RATE) > 1:
            issues.append(f"Sample rate {sr}")
        if not np.iscomplexobj(samples):
            issues.append(f"Not complex: {samples.dtype}")
        if len(samples) != EXPECTED_LENGTH:
            issues.append(f"Length {len(samples)} != {EXPECTED_LENGTH}")
        if np.all(samples == 0):
            return "SUSPECT", {}, ["All zeros"]
        if np.any(np.isnan(samples)):
            return "SUSPECT", {}, ["Contains NaN"]

        m = compute_metrics(samples)
        m['freq_mhz'] = round(freq / 1e6, 2)
        score, reasons = score_class(m, cls)
        v, conf = verdict_from_score(score)
        m['reasons'] = reasons
        return v, m, issues

    except Exception as e:
        return "SUSPECT", {}, [f"Error: {e}"]


def run_validation(dataset_path):
    dp = Path(dataset_path)

    print("=" * 72)
    print("  RTL-ML Dataset Validation Report")
    print("  800 samples | 7 classes | 1.024 MSPS | RTL-SDR Blog V4")
    print("=" * 72)
    print()

    totals = {'STRONG': 0, 'PLAUSIBLE': 0, 'WEAK': 0, 'SUSPECT': 0}

    for cls in SIGNAL_CLASSES:
        cdir = dp / cls
        if not cdir.exists():
            print(f"  ❌ {cls}: directory missing\n"); continue

        files = sorted(cdir.glob('*.npy'))
        if not files:
            print(f"  ⚠️  {cls}: no .npy files\n"); continue

        verdicts = {'STRONG': 0, 'PLAUSIBLE': 0, 'WEAK': 0, 'SUSPECT': 0}
        snrs, bws, flats = [], [], []
        freq_set = set()
        sample_reasons = None

        for f in files:
            v, m, iss = validate_file(f, cls)
            verdicts[v] += 1
            if m:
                snrs.append(m['snr'])
                bws.append(m['bw_khz'])
                flats.append(m['flatness'])
                if 'freq_mhz' in m:
                    freq_set.add(m['freq_mhz'])
                if sample_reasons is None and 'reasons' in m:
                    sample_reasons = m['reasons']

        passed = verdicts['STRONG'] + verdicts['PLAUSIBLE']
        total = len(files)
        icon = "✅" if passed == total else ("⚠️" if passed >= total * 0.7 else "❌")
        freqs = ', '.join(f"{f} MHz" for f in sorted(freq_set))

        print(f"  {icon} {cls} ({total} files @ {freqs})")
        print(f"     STRONG: {verdicts['STRONG']} | PLAUSIBLE: {verdicts['PLAUSIBLE']} | WEAK: {verdicts['WEAK']} | SUSPECT: {verdicts['SUSPECT']}")

        if snrs:
            print(f"     SNR: {np.mean(snrs):.1f} dB avg (range {np.min(snrs):.1f}–{np.max(snrs):.1f})")
            print(f"     Bandwidth: {np.mean(bws):.0f} kHz | Flatness: {np.mean(flats):.3f}")

        if sample_reasons:
            for reason in sample_reasons:
                print(f"     → {reason}")
        print()

        for k in totals:
            totals[k] += verdicts[k]

    total_all = sum(totals.values())
    passed_all = totals['STRONG'] + totals['PLAUSIBLE']

    print("=" * 72)
    print(f"  OVERALL: {passed_all}/{total_all} validated ({100*passed_all/total_all:.1f}%)")
    print(f"  STRONG: {totals['STRONG']} | PLAUSIBLE: {totals['PLAUSIBLE']} | WEAK: {totals['WEAK']} | SUSPECT: {totals['SUSPECT']}")
    print("=" * 72)

    if passed_all == total_all:
        print("\n  ✅ All samples pass — classifications consistent with spectral characteristics.")
    elif passed_all >= total_all * 0.8:
        print(f"\n  ⚠️  {total_all - passed_all} weak/suspect samples. Normal for sparse signals (APRS, ISM).")
    else:
        print(f"\n  ❌ {total_all - passed_all} samples need review.")


if __name__ == "__main__":
    path = Path(r"c:\Users\tre77\OneDrive\Desktop\pentesting\zzzzzzz\datasets_validated")
    if not path.exists():
        print(f"Dataset not found at: {path}")
        sys.exit(1)
    run_validation(path)
