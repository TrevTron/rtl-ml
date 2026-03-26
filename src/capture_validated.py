#!/usr/bin/env python3
"""
RTL-ML Validated Dataset Capture v2
Fixes from v1: DC offset removal, auto gain, signal quality gate, no dead satellites
"""
from rtlsdr import RtlSdr
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
import json
import random


# Per-frequency gain config. 'auto' lets the tuner AGC handle it (recommended).
# Override with integer dB if a specific frequency needs manual tuning.
SIGNAL_DEFS = [
    {'label': 'ISM_sensors',   'freq': 433.92e6, 'gain': 'auto', 'desc': '📡 ISM Sensors (433.92 MHz)'},
    {'label': 'FM_broadcast',  'freq': 98.7e6,   'gain': 'auto', 'desc': '📻 FM Radio (98.7 MHz)'},
    {'label': 'NOAA_weather',  'freq': 162.4e6,  'gain': 'auto', 'desc': '🌤️  NOAA Weather (162.4 MHz)'},
    {'label': 'pager',         'freq': 152.84e6, 'gain': 'auto', 'desc': '📟 Pager (152.84 MHz)'},
    {'label': 'APRS',          'freq': 144.39e6, 'gain': 'auto', 'desc': '📡 APRS (144.39 MHz)'},
    {'label': 'FRS_GMRS',     'freq': 462.5625e6,'gain': 'auto', 'desc': '📻 FRS/GMRS Radio (462.56 MHz)'},
    {'label': 'noise',         'freq': 145.0e6,  'gain': 'auto', 'desc': '📊 Noise Baseline'},
]

# Minimum SNR (dB) required to save a sample. Noise class is exempt.
MIN_SNR_DB = 6.0

# Settle time after tuning to a new frequency (seconds)
TUNE_SETTLE_TIME = 0.3


class ValidatedSignalCapture:
    def __init__(self, sample_rate=1.024e6):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.validation_results = {}
        print(f"SDR initialized")
        print(f"  Sample rate: {self.sample_rate/1e6:.3f} MSPS")

    def set_gain(self, gain):
        """Set gain - 'auto' for AGC or integer for manual dB."""
        if gain == 'auto':
            self.sdr.gain = 'auto'
            print(f"  Gain: auto (AGC)")
        else:
            self.sdr.gain = gain
            print(f"  Gain: {self.sdr.gain} dB")

    def capture_signal(self, frequency, duration=0.5):
        """Capture IQ samples with DC offset removal."""
        self.sdr.center_freq = frequency
        time.sleep(TUNE_SETTLE_TIME)

        # Flush stale samples from buffer
        _ = self.sdr.read_samples(1024)

        num_samples = int(self.sample_rate * duration)
        samples = self.sdr.read_samples(num_samples)

        # DC offset removal (critical fix from v1)
        samples = samples - np.mean(samples)

        return samples

    def compute_snr(self, samples):
        """Compute SNR as signal power vs noise floor in dB.
        Uses the ratio of occupied bandwidth power to median noise floor,
        not just max/mean which is fooled by DC spikes.
        """
        fft_power = np.abs(np.fft.fft(samples)) ** 2
        # Median is a robust noise floor estimator
        noise_floor = np.median(fft_power)
        # Signal power: 95th percentile (captures real signal, ignores spikes)
        signal_level = np.percentile(fft_power, 95)
        if noise_floor < 1e-20:
            return 0.0
        return float(10 * np.log10(signal_level / noise_floor))

    def estimate_bandwidth(self, samples):
        """Estimate occupied signal bandwidth at -10dB from peak."""
        fft_vals = np.fft.fftshift(np.fft.fft(samples))
        fft_power = np.abs(fft_vals) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))

        peak_power = np.max(fft_power)
        threshold = peak_power * 0.1  # -10 dB
        mask = fft_power > threshold
        bw_freqs = freqs[mask]
        if len(bw_freqs) < 2:
            return 0.0
        return float(bw_freqs[-1] - bw_freqs[0])

    def save_sample(self, samples, label, freq, snr_db, output_dir='datasets_validated'):
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{label}_{timestamp}.npy"
        filepath = os.path.join(output_dir, label, filename)

        data = {
            'samples': samples,
            'center_freq': freq,
            'sample_rate': self.sample_rate,
            'timestamp': timestamp,
            'label': label,
            'duration': len(samples) / self.sample_rate,
            'snr_db': snr_db,
            'version': 'v2',
        }

        np.save(filepath, data)

    def generate_spectrogram(self, samples, label, freq, output_dir='visualizations'):
        """Generate spectrogram + PSD plot. Samples should already be DC-removed."""
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Spectrogram
        plt.subplot(2, 1, 1)
        f, t, Sxx = signal.spectrogram(samples, fs=self.sample_rate, nperseg=1024)
        plt.pcolormesh(t, f/1e6, 10*np.log10(Sxx + 1e-20), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency (MHz)')
        plt.xlabel('Time (s)')
        plt.title(f'{label} - Spectrogram - {freq/1e6:.2f} MHz')
        plt.colorbar(label='Power (dB)')

        # PSD
        plt.subplot(2, 1, 2)
        fft_vals = np.fft.fftshift(np.fft.fft(samples))
        fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/self.sample_rate))
        plt.plot(fft_freqs/1e6, 10*np.log10(np.abs(fft_vals)**2 + 1e-20))
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency Offset (MHz)')
        plt.title(f'{label} - Power Spectral Density (DC removed)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f'{label}_spectrum.png')
        plt.savefig(filepath, dpi=150)
        plt.close()

        return filepath

    # ── Per-class validators ──────────────────────────────────────────

    def validate_adsb(self, samples):
        power = np.abs(samples) ** 2
        burst_ratio = np.max(power) / (np.mean(power) + 1e-10)
        return {
            'pass': burst_ratio > 10,
            'burst_ratio': float(burst_ratio),
            'note': 'ADS-B: expect >10x burst ratio from transponder pulses'
        }

    def validate_ism(self, samples):
        power = np.abs(samples) ** 2
        burst_ratio = np.max(power) / (np.mean(power) + 1e-10)
        return {
            'pass': burst_ratio > 5,
            'burst_ratio': float(burst_ratio),
            'note': 'ISM: expect >5x burst ratio from sensor transmissions'
        }

    def validate_fm(self, samples):
        bw = self.estimate_bandwidth(samples)
        return {
            'pass': bw > 50e3,
            'bandwidth_khz': float(bw / 1e3),
            'note': f'FM: expect >50 kHz occupied BW (got {bw/1e3:.0f} kHz)'
        }

    def validate_weather(self, samples):
        snr = self.compute_snr(samples)
        return {
            'pass': snr > 6,
            'snr_db': snr,
            'note': f'NOAA Weather: expect >6 dB SNR (got {snr:.1f} dB)'
        }

    def validate_aprs(self, samples):
        """APRS: expect bursty packet with >10 kHz BW or >5x burst ratio."""
        bw = self.estimate_bandwidth(samples)
        power = np.abs(samples) ** 2
        burst_ratio = np.max(power) / (np.mean(power) + 1e-10)
        return {
            'pass': bw > 10e3 or burst_ratio > 5,
            'bandwidth_khz': float(bw / 1e3),
            'burst_ratio': float(burst_ratio),
            'note': f'APRS: BW={bw/1e3:.0f}kHz burst={burst_ratio:.1f}x (need >10kHz or >5x)'
        }

    def validate_frs_gmrs(self, samples):
        """FRS/GMRS: expect NFM voice with >5 kHz BW and >6 dB SNR."""
        bw = self.estimate_bandwidth(samples)
        snr = self.compute_snr(samples)
        return {
            'pass': bw > 5e3 and snr > MIN_SNR_DB,
            'bandwidth_khz': float(bw / 1e3),
            'snr_db': snr,
            'note': f'FRS/GMRS: BW={bw/1e3:.0f}kHz SNR={snr:.1f}dB (need >5kHz BW)'
        }

    def validate_pager(self, samples):
        """Pager: expect FSK modulation with >8 kHz bandwidth."""
        bw = self.estimate_bandwidth(samples)
        snr = self.compute_snr(samples)
        return {
            'pass': bw > 8e3 and snr > MIN_SNR_DB,
            'bandwidth_khz': float(bw / 1e3),
            'snr_db': snr,
            'note': f'Pager: BW={bw/1e3:.0f}kHz SNR={snr:.1f}dB (need >8kHz BW)'
        }

    def validate_signal(self, samples, label):
        """Route to class-specific validator. Returns dict with 'pass' key."""
        if label == 'ISM_sensors':
            return self.validate_ism(samples)
        elif label == 'FM_broadcast':
            return self.validate_fm(samples)
        elif label == 'NOAA_weather':
            return self.validate_weather(samples)
        elif label == 'APRS':
            return self.validate_aprs(samples)
        elif label == 'pager':
            return self.validate_pager(samples)
        elif label == 'FRS_GMRS':
            return self.validate_frs_gmrs(samples)
        elif label == 'noise':
            return {'pass': True, 'note': 'Noise baseline (always passes)'}
        else:
            # Generic: just measure SNR
            snr = self.compute_snr(samples)
            return {
                'pass': snr > MIN_SNR_DB,
                'snr_db': snr,
                'note': f'Generic SNR check: {snr:.1f} dB (min {MIN_SNR_DB})'
            }

    def close(self):
        self.sdr.close()


def main():
    print("=" * 70)
    print("RTL-ML VALIDATED DATASET CAPTURE  v2")
    print("DC removal | Auto gain | Signal quality gate")
    print("=" * 70)

    signals = SIGNAL_DEFS
    samples_per_class = 100

    print(f"\n📊 Capture Plan:")
    print(f"   {len(signals)} signal classes, {samples_per_class} samples each")
    print(f"   DC offset removal: ON")
    print(f"   Gain: per-frequency (auto AGC)")
    print(f"   Quality gate: SNR > {MIN_SNR_DB} dB + per-class validation")
    print(f"   Temporal spacing: 2s between captures")
    print(f"   Total: {len(signals) * samples_per_class} target samples\n")

    capture = ValidatedSignalCapture()
    validation_summary = {}
    capture_stats = {}

    # Randomize class order to prevent SDR thermal warm-up bias
    random.shuffle(signals)

    for sig in signals:
        label = sig['label']
        freq = sig['freq']
        gain = sig['gain']
        desc = sig['desc']

        print(f"\n{'─'*60}")
        print(f"{desc}")
        print(f"   Frequency: {freq/1e6:.2f} MHz")
        capture.set_gain(gain)

        # ── Quick lock check: capture one sample and verify tuner locked ──
        print(f"   🔒 Checking tuner lock...")
        test_samples = capture.capture_signal(freq, duration=0.2)
        test_snr = capture.compute_snr(test_samples)
        test_bw = capture.estimate_bandwidth(test_samples)
        print(f"      SNR: {test_snr:.1f} dB | BW: {test_bw/1e3:.0f} kHz")

        # ── Capture samples with SNR gate + per-class validation ──
        print(f"   📦 Capturing {samples_per_class} samples...")
        saved = 0
        rejected = 0
        max_attempts = samples_per_class * 5  # generous retries for strict validation

        pbar = tqdm(total=samples_per_class, desc="   Progress")
        attempts = 0
        while saved < samples_per_class and attempts < max_attempts:
            attempts += 1
            samples = capture.capture_signal(freq)
            snr = capture.compute_snr(samples)

            # SNR quality gate (noise class is exempt)
            if label != 'noise' and snr < MIN_SNR_DB:
                rejected += 1
                continue

            # Per-class signal validation on EVERY sample before saving
            validation = capture.validate_signal(samples, label)
            if not validation.get('pass', False):
                rejected += 1
                continue

            capture.save_sample(samples, label, freq, snr)
            saved += 1
            pbar.update(1)

            # Temporal spacing to reduce same-moment correlation
            time.sleep(2)
        pbar.close()

        if saved < samples_per_class:
            print(f"   ⚠️  Only captured {saved}/{samples_per_class} ({rejected} rejected)")
        else:
            print(f"   ✅ {saved} samples saved ({rejected} rejected by quality gate)")

        # ── Spectrogram (fresh capture for clean visualization) ──
        print(f"   📊 Generating spectrogram...")
        vis_samples = capture.capture_signal(freq, duration=1.0)
        spec_path = capture.generate_spectrogram(vis_samples, label, freq)

        # ── Class summary ──
        final_validation = capture.validate_signal(vis_samples, label)
        validation_summary[label] = final_validation
        pass_rate = saved / max(attempts, 1) * 100
        status = "✅" if saved == samples_per_class else "⚠️"
        print(f"      {status} {saved}/{samples_per_class} saved, {rejected} rejected ({pass_rate:.0f}% pass rate)")
        if not final_validation.get('pass', False):
            print(f"      ⚠️  Final validation: {final_validation.get('note', '')}")

        capture_stats[label] = {
            'saved': saved,
            'rejected': rejected,
            'attempts': attempts,
            'pass_rate': pass_rate,
            'validation': final_validation,
        }

    capture.close()

    # ── Save reports ──
    report = {
        'version': 'v2',
        'capture_date': datetime.now().isoformat(),
        'sample_rate': 1.024e6,
        'gain_mode': 'per-frequency auto AGC',
        'dc_removal': True,
        'snr_gate_db': MIN_SNR_DB,
        'signals': capture_stats,
        'validation': validation_summary,
    }
    with open('validation_report_v2.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("✅ V2 CAPTURE COMPLETE")
    print("=" * 70)

    total_saved = sum(s['saved'] for s in capture_stats.values())
    total_rejected = sum(s['rejected'] for s in capture_stats.values())

    print(f"\n📁 Output:")
    print(f"   datasets_validated/       {total_saved} samples ({total_rejected} rejected)")
    print(f"   visualizations/           {len(signals)} spectrograms")
    print(f"   validation_report_v2.json capture metadata")
    print(f"\n📊 Per-class summary:")
    for label, stats in capture_stats.items():
        v = stats['validation']
        status = "✅" if v.get('pass') else "⚠️"
        print(f"   {status} {label:15s}  saved={stats['saved']:3d}  rejected={stats['rejected']:3d}  pass_rate={stats.get('pass_rate', 0):.0f}%")

    # Check for any failures
    failed = [l for l, s in capture_stats.items() if not s['validation'].get('pass')]
    if failed:
        print(f"\n⚠️  Classes with marginal validation: {', '.join(failed)}")
        print(f"   Check antenna placement and retry if needed.")
    else:
        print(f"\n🎯 All classes passed validation!")


if __name__ == '__main__':
    main()
