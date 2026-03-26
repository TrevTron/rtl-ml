#!/usr/bin/env python3
"""Generate publication-quality spectrograms for all 7 v2 signal classes."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sig
import os
import glob

base = os.path.expanduser('~/rtl-ml/datasets_validated')
out_dir = os.path.expanduser('~/rtl-ml/spectrograms_v2')
os.makedirs(out_dir, exist_ok=True)

classes = sorted(os.listdir(base))

# Color scheme
cmap = 'viridis'

for cls in classes:
    files = sorted(glob.glob(os.path.join(base, cls, '*.npy')))
    if not files:
        continue
    
    # Load a representative sample (pick one from the middle)
    mid = len(files) // 2
    data = np.load(files[mid], allow_pickle=True).item()
    samples = data['samples'] - np.mean(data['samples'])
    freq_mhz = data['center_freq'] / 1e6
    sr = data['sample_rate']
    snr = data.get('snr_db', 0)
    
    # Compute spectrogram
    f, t, Sxx = sig.spectrogram(samples, fs=sr, nperseg=1024, 
                                 noverlap=512, return_onesided=False)
    # Shift to center frequency
    f_shifted = np.fft.fftshift(f)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx_shifted + 1e-30)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Spectrogram
    ax1 = axes[0]
    im = ax1.pcolormesh(t * 1000, (f_shifted / 1000) + freq_mhz * 1000, 
                         Sxx_db, cmap=cmap, shading='gouraud')
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(f'{cls} — {freq_mhz:.2f} MHz (SNR: {snr:.1f} dB)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Power (dB)')
    
    # PSD plot
    ax2 = axes[1]
    freqs_psd, psd = sig.welch(samples, fs=sr, nperseg=4096, return_onesided=False)
    freqs_psd_shifted = np.fft.fftshift(freqs_psd)
    psd_shifted = np.fft.fftshift(psd)
    ax2.semilogy((freqs_psd_shifted / 1000) + freq_mhz * 1000, psd_shifted)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('PSD')
    ax2.set_title('Power Spectral Density')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{cls}_spectrogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')

# Also create a combined overview figure
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

for i, cls in enumerate(classes):
    files = sorted(glob.glob(os.path.join(base, cls, '*.npy')))
    if not files or i >= 7:
        continue
    
    mid = len(files) // 2
    data = np.load(files[mid], allow_pickle=True).item()
    samples = data['samples'] - np.mean(data['samples'])
    sr = data['sample_rate']
    freq_mhz = data['center_freq'] / 1e6
    
    f, t, Sxx = sig.spectrogram(samples, fs=sr, nperseg=1024,
                                 noverlap=512, return_onesided=False)
    f_shifted = np.fft.fftshift(f)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx_shifted + 1e-30)
    
    ax = axes[i]
    ax.pcolormesh(t * 1000, f_shifted / 1000, Sxx_db, cmap=cmap, shading='gouraud')
    ax.set_title(f'{cls}\n({freq_mhz:.2f} MHz)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=8)
    ax.set_ylabel('Freq offset (kHz)', fontsize=8)

# Hide extra subplot
if len(classes) < 8:
    axes[7].set_visible(False)

fig.suptitle('RTL-ML v2 Dataset — Signal Spectrograms (7 Classes)', fontsize=16, fontweight='bold')
plt.tight_layout()
combined_path = os.path.join(out_dir, 'all_classes_overview.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {combined_path}')

print(f'\nAll spectrograms saved to {out_dir}/')
