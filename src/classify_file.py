#!/usr/bin/env python3
"""
Classify IQ recordings stored on disk.

Reads captures from gqrx, RFAnalyzer, SDR#/SDR++ and rtl_sdr (raw complex
float32/int16/uint8, WAV or the project's .npy format), removes the DC offset,
resamples to the 1.024 MSPS training rate, splits the recording into 0.5 s
windows and -- with --classify -- reports a class per window and a majority
verdict.

    python src/classify_file.py capture.iq
    python src/classify_file.py --classify samples/*.raw
    python src/classify_file.py --classify --no-resample capture.iq
"""
import argparse
import os
import pickle
import re
import sys
import warnings
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_features import SignalFeatureExtractor

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except Exception:
    pass

TRAIN_RATE = 1.024e6  # sample rate used to train the model
WINDOW_S = 0.5        # training window length
FORMATS = ('auto', 'cf32', 'cs16', 'cu8', 'wav', 'npy')
RAW_DTYPES = {'cf32': np.float32, 'cs16': np.int16, 'cu8': np.uint8}
EXT_FORMAT = {
    '.cf32': 'cf32', '.fc32': 'cf32', '.cfile': 'cf32',
    '.cs16': 'cs16', '.sc16': 'cs16', '.complex16s': 'cs16',
    '.cu8': 'cu8', '.c8': 'cu8',
    '.wav': 'wav', '.wave': 'wav', '.npy': 'npy', '.npz': 'npy',
}
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'models', 'rtl_classifier_validated.pkl')


def metadata_from_name(name):
    """Recover (center_freq_hz, sample_rate_hz) from common capture filenames.

    gqrx I/Q:   gqrx_YYYYMMDD_HHMMSS_<freq>_<rate>_fc.raw
    gqrx audio: gqrx_YYYYMMDD_HHMMSS_<freq>.wav        (rate is in the WAV header)
    RFAnalyzer/SDR#: ..._<freq>Hz_<rate>Sps.iq
    """
    freq = rate = None
    m = re.search(r'gqrx[_-]\d{8}[_-]\d{6}[_-](\d{4,})(?:[_-](\d{3,}))?', name)
    if m:
        freq = float(m.group(1))
        rate = float(m.group(2)) if m.group(2) else None
    scales = {'': 1, 'k': 1e3, 'm': 1e6, 'g': 1e9}
    m = re.search(r'(\d+(?:\.\d+)?)\s*([kKmMgG]?)Sps', name)
    if m:
        rate = float(m.group(1)) * scales[m.group(2).lower()]
    m = re.search(r'(\d+(?:\.\d+)?)\s*([kKmMgG]?)Hz', name)
    if m:
        freq = float(m.group(1)) * scales[m.group(2).lower()]
    return freq, rate


def sniff_format(path):
    """Guess cf32/cs16/cu8 from the first 64 KB of an extension-less capture."""
    head = np.fromfile(path, dtype=np.uint8, count=65536)
    floats = head[:len(head) // 4 * 4].view(np.float32)
    if np.all(np.isfinite(floats)) and np.max(np.abs(floats)) < 1e3 \
            and np.mean(np.abs(floats) > 1e-4) > 0.3:
        return 'cf32'
    return 'cu8' if np.mean((head >= 96) & (head <= 160)) > 0.6 else 'cs16'


def read_iq(path, fmt):
    """Return (complex64 samples, embedded sample rate or None, format used)."""
    ext = os.path.splitext(path)[1].lower()
    if fmt == 'auto':
        fmt = EXT_FORMAT.get(ext) or sniff_format(path)

    if fmt == 'npy':
        obj = np.load(path, allow_pickle=True)
        obj = obj.item() if obj.dtype == object and obj.shape == () else obj
        if isinstance(obj, dict):
            return np.asarray(obj['samples'], np.complex64), obj.get('sample_rate'), fmt
        return np.asarray(obj, np.complex64), None, fmt

    if fmt == 'wav':
        rate, data = wavfile.read(path)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        samples = data[:, 0] + 1j * data[:, 1] if data.ndim == 2 else data + 0j
        return samples.astype(np.complex64), rate, fmt

    data = np.fromfile(path, dtype=RAW_DTYPES[fmt])
    data = data[:len(data) // 2 * 2].astype(np.float32)
    if fmt == 'cf32':
        return data.view(np.complex64), None, fmt
    data = data / 32768.0 if fmt == 'cs16' else data / 127.5 - 1.0
    return (data[0::2] + 1j * data[1::2]).astype(np.complex64), None, fmt


def to_train_rate(samples, rate):
    """Resample to the training rate unless already there."""
    if abs(rate - TRAIN_RATE) < 1:
        return samples, rate
    ratio = Fraction(int(round(TRAIN_RATE)), int(round(rate))).limit_denominator(1000)
    out = signal.resample_poly(samples, ratio.numerator, ratio.denominator)
    return out.astype(np.complex64), float(TRAIN_RATE)


def majority(results, classes):
    """Most frequent class; ties broken by summed probability. Returns (name, votes)."""
    counts = {}
    for name, _ in results:
        counts[name] = counts.get(name, 0) + 1
    top = max(counts.values())
    tied = [name for name, count in counts.items() if count == top]
    if len(tied) == 1:
        return tied[0], top
    score = {name: sum(p[classes.index(name)] for _, p in results) for name in tied}
    return max(score, key=score.get), top


def run(path, args, extractor, model_data):
    samples, embedded_rate, fmt = read_iq(path, args.format)
    name_freq, name_rate = metadata_from_name(os.path.basename(path))
    rate = args.sample_rate or embedded_rate or name_rate or TRAIN_RATE
    freq = args.freq if args.freq is not None else name_freq
    source_rate = rate

    samples = samples - samples.mean()  # DC removal, as in the training capture
    if args.resample:
        samples, rate = to_train_rate(samples, rate)

    window = max(1, int(round(rate * WINDOW_S)))
    windows = [samples[i:i + window] for i in range(0, len(samples) - window + 1, window)]
    windows = windows or [samples]

    print("=" * 72)
    print(f"FILE: {path}")
    print(f"  format      : {fmt}")
    rate_line = f"  sample rate : {rate / 1e6:.6g} MSPS"
    if rate != source_rate:
        rate_line += f" (resampled from {source_rate / 1e6:.6g} MSPS)"
    print(rate_line)
    print(f"  center freq : {freq / 1e6:.7g} MHz" if freq else "  center freq : unknown")
    print(f"  samples     : {len(samples):,} ({len(samples) / rate:.3f} s, "
          f"{len(windows)} window(s))")

    features = [extractor.extract_features(w) for w in windows]
    print("  features (mean per window):")
    for name, value in zip(extractor.feature_names, np.mean(features, axis=0)):
        print(f"      {name:16s} {value: .5e}")

    if not args.classify:
        return

    scaler, clf = model_data['scaler'], model_data['model']
    classes = [str(c) for c in clf.classes_]
    results = []
    print(f"  classifier ({model_data.get('model_name', '?')}):")
    for i, feature in enumerate(features):
        probs = clf.predict_proba(scaler.transform(feature.reshape(1, -1)))[0]
        name = classes[int(np.argmax(probs))]
        results.append((name, probs))
        print(f"      #{i:<3d} {name:15s} {probs.max() * 100:5.1f}%")

    winner, votes = majority(results, classes)
    print(f"      => {winner}  ({votes}/{len(results)} windows)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog='formats: cf32 (gqrx/SDR# float32), cs16 (.cs16/.complex16s), '
               'cu8 (rtl_sdr/RFAnalyzer), wav, npy (dataset captures).\n'
               'Off-rate recordings are resampled to 1.024 MSPS unless '
               '--no-resample is given.',
    )
    parser.add_argument('files', nargs='+', help='IQ recording(s)')
    parser.add_argument('--format', choices=FORMATS, default='auto',
                        help='input sample format (default: auto-detect)')
    parser.add_argument('--sample-rate', type=float, help='file sample rate in Hz')
    parser.add_argument('--freq', type=float, help='center frequency in Hz (metadata)')
    parser.add_argument('--classify', action='store_true',
                        help='run the trained classifier (default: features only)')
    parser.add_argument('--model', help='model .pkl (default: models/rtl_classifier_validated.pkl)')
    parser.add_argument('--no-resample', dest='resample', action='store_false',
                        help='analyze at the native sample rate')
    args = parser.parse_args(argv)

    extractor = SignalFeatureExtractor(sample_rate=TRAIN_RATE)
    if args.classify:
        try:
            with open(args.model or DEFAULT_MODEL, 'rb') as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            parser.error(f"model not found: {args.model or DEFAULT_MODEL}")
    else:
        model_data = None

    for path in args.files:
        run(path, args, extractor, model_data)


if __name__ == '__main__':
    main()
