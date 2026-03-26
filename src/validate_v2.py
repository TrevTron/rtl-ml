#!/usr/bin/env python3
"""V2 Validation: Visual & Feature checks before publishing."""
import numpy as np
import os
import glob
from scipy import signal as sig

base = os.path.expanduser('~/rtl-ml/datasets_validated')
classes = sorted(os.listdir(base))

# ── 1. SPECTROGRAM VISUAL CHECK ──────────────────────────────────────
print("=" * 70)
print("CHECK 1: SPECTROGRAM VISUAL DIFFERENCES")
print("=" * 70)
print("Computing spectral characteristics for each class...\n")

class_profiles = {}
for cls in classes:
    files = sorted(glob.glob(os.path.join(base, cls, '*.npy')))
    if not files:
        continue
    
    bws = []
    peak_freqs = []
    spectral_flatness_vals = []
    power_spreads = []
    
    for f in files[:20]:  # sample 20 per class
        data = np.load(f, allow_pickle=True).item()
        s = data['samples'] - np.mean(data['samples'])  # DC removal
        
        # Power spectral density
        freqs, psd = sig.welch(s, fs=1024000, nperseg=4096)
        
        # Bandwidth (where PSD > 10% of peak)
        psd_norm = psd / psd.max()
        bw_bins = np.sum(psd_norm > 0.1)
        bw_khz = bw_bins * (1024000 / 2) / len(freqs) / 1000
        bws.append(bw_khz)
        
        # Peak frequency offset from center
        peak_idx = np.argmax(psd)
        peak_freq_khz = (freqs[peak_idx] - 512000) / 1000
        peak_freqs.append(peak_freq_khz)
        
        # Spectral flatness (1.0 = white noise, 0.0 = pure tone)
        geo_mean = np.exp(np.mean(np.log(psd + 1e-30)))
        arith_mean = np.mean(psd)
        flatness = geo_mean / (arith_mean + 1e-30)
        spectral_flatness_vals.append(flatness)
        
        # Power spread (std of PSD)
        power_spreads.append(np.std(psd))
    
    profile = {
        'bw_mean': np.mean(bws),
        'bw_std': np.std(bws),
        'peak_freq': np.mean(peak_freqs),
        'flatness': np.mean(spectral_flatness_vals),
        'power_spread': np.mean(power_spreads),
    }
    class_profiles[cls] = profile
    
    print(f"  {cls:20s}  BW={profile['bw_mean']:6.1f}+/-{profile['bw_std']:4.1f} kHz  "
          f"Flatness={profile['flatness']:.4f}  PowerSpread={profile['power_spread']:.6f}")

# Check if classes look different
print("\n  PAIRWISE DIFFERENCES:")
fm = class_profiles.get('FM_broadcast', {})
noise = class_profiles.get('noise', {})
if fm and noise:
    bw_ratio = fm.get('bw_mean', 0) / max(noise.get('bw_mean', 1), 0.01)
    flat_diff = abs(fm.get('flatness', 0) - noise.get('flatness', 0))
    print(f"  FM vs Noise: BW ratio={bw_ratio:.1f}x, Flatness diff={flat_diff:.4f}")
    if bw_ratio > 3:
        print("  ✅ FM clearly wider than noise")
    else:
        print("  ❌ FM NOT clearly wider than noise")
    if flat_diff > 0.01:
        print("  ✅ Spectral shapes clearly different")
    else:
        print("  ⚠️  Spectral shapes similar")

# Check uniqueness across all classes
all_bws = [p['bw_mean'] for p in class_profiles.values()]
all_flat = [p['flatness'] for p in class_profiles.values()]
bw_cv = np.std(all_bws) / max(np.mean(all_bws), 0.01)
flat_cv = np.std(all_flat) / max(np.mean(all_flat), 0.01)
print(f"\n  BW coefficient of variation across classes: {bw_cv:.2f}")
print(f"  Flatness coefficient of variation: {flat_cv:.2f}")
if bw_cv > 0.3:
    print("  ✅ Classes have diverse bandwidths (not all identical)")
else:
    print("  ❌ Classes have similar bandwidths (v1 problem)")

# ── 2. FEATURE SANITY CHECK ──────────────────────────────────────────
print("\n" + "=" * 70)
print("CHECK 2: 18-FEATURE SANITY CHECK")
print("=" * 70)
print("Extracting features for each class...\n")

def extract_18_features(samples):
    features = []
    power = np.abs(samples) ** 2
    features.append(np.mean(power))        # 0: power_mean
    features.append(np.std(power))         # 1: power_std
    features.append(np.max(power))         # 2: power_max
    features.append(np.min(power))         # 3: power_min
    
    fft_vals = np.fft.fft(samples)
    fft_power = np.abs(fft_vals) ** 2
    features.append(np.mean(fft_power))    # 4: fft_mean
    features.append(np.std(fft_power))     # 5: fft_std
    features.append(np.max(fft_power))     # 6: fft_max
    features.append(np.argmax(fft_power) / len(fft_power))  # 7: fft_peak_idx
    
    i_s = np.real(samples)
    q_s = np.imag(samples)
    features.append(np.mean(i_s))          # 8: i_mean
    features.append(np.std(i_s))           # 9: i_std
    features.append(np.mean(q_s))          # 10: q_mean
    features.append(np.std(q_s))           # 11: q_std
    
    phase = np.angle(samples)
    features.append(np.mean(phase))        # 12: phase_mean
    features.append(np.std(phase))         # 13: phase_std
    
    phase_diff = np.diff(phase)
    features.append(np.mean(phase_diff))   # 14: phase_diff_mean
    features.append(np.std(phase_diff))    # 15: phase_diff_std
    
    bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
    features.append(bandwidth / len(fft_power))  # 16: bw_ratio
    
    return np.array(features)

feature_names = [
    'power_mean', 'power_std', 'power_max', 'power_min',
    'fft_mean', 'fft_std', 'fft_max', 'fft_peak_idx',
    'i_mean', 'i_std', 'q_mean', 'q_std',
    'phase_mean', 'phase_std', 'phase_diff_mean', 'phase_diff_std',
    'bw_ratio'
]

class_features = {}
for cls in classes:
    files = sorted(glob.glob(os.path.join(base, cls, '*.npy')))
    if not files:
        continue
    feats = []
    for f in files[:20]:
        data = np.load(f, allow_pickle=True).item()
        s = data['samples'] - np.mean(data['samples'])
        feats.append(extract_18_features(s))
    class_features[cls] = np.mean(feats, axis=0)

# Print feature comparison table
print(f"{'Feature':20s}", end='')
for cls in classes:
    print(f"  {cls[:8]:>10s}", end='')
print()
print("-" * (20 + 12 * len(classes)))

for i, fname in enumerate(feature_names):
    vals = [class_features[cls][i] for cls in classes]
    cv = np.std(vals) / max(abs(np.mean(vals)), 1e-30)
    print(f"{fname:20s}", end='')
    for cls in classes:
        print(f"  {class_features[cls][i]:10.4g}", end='')
    flag = " ✅" if cv > 0.2 else " ⚠️"
    print(f"  CV={cv:.2f}{flag}")

# Count how many features actually discriminate
discriminating = 0
for i, fname in enumerate(feature_names):
    vals = [class_features[cls][i] for cls in classes]
    cv = np.std(vals) / max(abs(np.mean(vals)), 1e-30)
    if cv > 0.2:
        discriminating += 1

print(f"\n  {discriminating}/{len(feature_names)} features have meaningful variation across classes")
if discriminating >= 10:
    print("  ✅ Strong feature discrimination")
elif discriminating >= 5:
    print("  ⚠️  Moderate feature discrimination")
else:
    print("  ❌ Weak feature discrimination (v1 problem)")

print("\n✅ VALIDATION CHECKS COMPLETE")
