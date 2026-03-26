#!/usr/bin/env python3
"""Check FM audio capture for real content."""
import numpy as np

d = np.fromfile('/tmp/fm_audio.raw', dtype=np.int16)
print(f'Samples: {len(d)}')
print(f'Std: {d.std():.1f}')
print(f'Range: [{d.min()}, {d.max()}]')

# Check for audio content - real audio has varying amplitude
chunks = np.array_split(d, 10)
rms_vals = [np.sqrt(np.mean(c.astype(float)**2)) for c in chunks]
rms_strs = [f"{r:.0f}" for r in rms_vals]
print(f'RMS per chunk: {rms_strs}')
rms_cv = np.std(rms_vals) / np.mean(rms_vals)
print(f'RMS variation: {rms_cv:.3f}')

# Zero crossing rate
zc = np.sum(np.diff(np.sign(d.astype(float))) != 0) / (len(d) / 48000)
print(f'Zero crossing rate: {zc:.0f}/s')

if d.std() > 1000 and rms_cv > 0.05:
    print('VERDICT: REAL FM AUDIO (music/speech detected)')
elif d.std() > 500:
    print('VERDICT: FM signal present (possibly weak station)')
elif d.std() > 50:
    print('VERDICT: Weak signal or static')
else:
    print('VERDICT: NO SIGNAL')
