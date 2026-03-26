#!/usr/bin/env python3
"""Quick quality check on captured v2 dataset."""
import numpy as np
import os
import glob

base = os.path.expanduser('~/rtl-ml/datasets_validated')

print('Class                Files  I_std     SNR_dB    BW_kHz')
print('-' * 65)

for cls in sorted(os.listdir(base)):
    files = sorted(glob.glob(os.path.join(base, cls, '*.npy')))
    if not files:
        continue
    stds = []
    snrs = []
    bws = []
    for f in files[:10]:  # sample first 10
        data = np.load(f, allow_pickle=True).item()
        s = data['samples'] - np.mean(data['samples'])
        stds.append(s.real.std())
        snrs.append(data.get('snr_db', 0))
        fft = np.abs(np.fft.fft(s))
        bw_bins = np.sum(fft > fft.mean() * 3)
        bws.append(bw_bins * 1024000 / len(fft) / 1000)
    
    avg_std = np.mean(stds)
    avg_snr = np.mean(snrs)
    avg_bw = np.mean(bws)
    print(f'{cls:20s}  {len(files):5d}  {avg_std:8.4f}  {avg_snr:8.1f}  {avg_bw:8.0f}')

print()
total = sum(len(glob.glob(os.path.join(base, c, '*.npy'))) for c in os.listdir(base))
print(f'Total samples: {total}')
