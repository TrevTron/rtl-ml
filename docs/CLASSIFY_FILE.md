# Classifying IQ Recordings on Disk

`src/classify_file.py` reads a capture from disk, extracts the same 17 features as
the live classifier and — with `--classify` — labels it. It reuses
`SignalFeatureExtractor` and the trained Random Forest model, so a file and a live
capture of the same signal produce the same features.

It handles captures from **gqrx**, **RFAnalyzer**, **SDR#/SDR++** and **rtl_sdr**
(raw complex float32/int16/uint8, WAV, or the project's `.npy` format).

## Quick start

```bash
# Features only
python src/classify_file.py capture.iq

# Features + classification
python src/classify_file.py --classify capture.iq

# Classify everything the shell expands
python src/classify_file.py --classify samples/*.raw
```

```
FILE: samples/gqrx_20261007_175600_413016225_1000000_fc.raw
  format      : cf32
  sample rate : 1.024 MSPS (resampled from 1 MSPS)
  center freq : 413.0162 MHz
  samples     : 1,600,000 (1.600 s, 3 window(s))
  features (mean per window):
      power_mean        5.65600e-01
      ...
  classifier (Random Forest):
      #0   FM_broadcast     95.0%
      #1   FM_broadcast     94.0%
      #2   FM_broadcast     94.0%
      => FM_broadcast  (3/3 windows)
```

## What it does to a recording

1. **Read** the samples (format auto-detected from the extension, then the file
   contents for bare `.raw`/`.iq` files).
2. **Remove the DC offset** (`samples -= mean`), matching the training capture.
3. **Resample to 1.024 MSPS** if the file was captured at another rate.
4. **Split into 0.5 s windows** (the training window length).
5. **Extract features** and, with `--classify`, label each window and report the
   majority verdict.

### Sample rate

Center frequency is metadata only, but the sample rate drives the window size.
The model was trained on 0.5 s windows at **1.024 MSPS**; the features are
rate-sensitive, so an off-rate file (e.g. 1.8 or 2 MSPS) is systematically
misclassified unless it is resampled. `classify_file.py` therefore resamples to
1.024 MSPS by default (`scipy.signal.resample_poly`). Use `--no-resample` to
analyze at the native rate.

The rate and center frequency are read from the filename or the WAV/npy header:

| File name | Detected |
|-----------|----------|
| `gqrx_20261007_175600_413016225_1000000_fc.raw` | 413.016225 MHz, 1000000 SPS (gqrx I/Q) |
| `gqrx_20261008_012726_433092000.wav` | 433.092 MHz; rate from the WAV header (gqrx audio) |
| `..._432999500Hz_2MSps.iq` | 432.9995 MHz, 2 MSPS |
| `capture.iq` | unknown → pass `--sample-rate 2e6` |

Date (`YYYYMMDD`) and time (`HHMMSS`) fields in filenames are not mistaken for
frequencies or rates. Override anything with `--sample-rate 2e6` / `--freq 162.4e6`.

### Files longer than 0.5 s

Long recordings are split into consecutive 0.5 s windows so a burst is not lost in
a long idle stretch. Every window is classified and the file's verdict is the
**majority** class (ties broken by summed probability); the per-window labels are
printed too, which is often more informative for bursty signals.

## Options

```
--format {auto,cf32,cs16,cu8,wav,npy}   input sample format (default: auto)
--sample-rate HZ                        file sample rate (default: filename/header)
--freq HZ                               center frequency (metadata only)
--classify                              run the trained classifier
--model PATH                            model .pkl (default: models/rtl_classifier_validated.pkl)
--no-resample                           analyze at the native sample rate
```

## Formats

| `--format` | Layout | Extensions | Produced by |
|------------|--------|------------|-------------|
| `cf32` | complex float32 | `.cf32` `.raw` `.iq` | gqrx I/Q, SDR#/SDR++ |
| `cs16` | complex int16 | `.cs16` `.complex16s` | SDR++, some Android apps |
| `cu8` | complex uint8 | `.cu8` `.c8` `.raw` `.iq` | `rtl_sdr`, RFAnalyzer raw |
| `wav` | 16-bit PCM stereo (L=I, R=Q) | `.wav` | RFAnalyzer, gqrx audio |
| `npy` | numpy capture (dict with `samples`) | `.npy` | `capture_validated.py` |

`auto` resolves from the extension; for a bare `.raw`/`.iq` it inspects the first
64 KB (float32 vs. int16 vs. uint8). Pass `--format` when in doubt.

gqrx `.raw` I/Q recordings are `cf32_le` (confirmed against the gqrx source:
`gr::blocks::file_sink` writing `gr_complex`). gqrx **audio** recordings are WAV
named `gqrx_YYYYMMDD_HHMMSS_<freq>.wav` (no rate in the name).

## Downsampling 2 MSPS IQ to the training rate

The tool already does this automatically, so the simplest command is just:

```bash
python src/classify_file.py --classify 2MSps.iq
```

To write a resampled copy for other tooling, any of these work — set rate,
element type and channel count correctly or the I/Q stream will be misread:

```bash
# SoX (resamples each of the two channels = correct complex resampling)
sox -t raw -r 2000000 -e floating-point   -b 32 -c 2 -L 2MSps.iq  \
    -t raw -r 1024000 -e floating-point   -b 32 -c 2 -L 1.024MSps.iq rate
sox -t raw -r 2000000 -e unsigned-integer -b 8  -c 2 -L 2MSps.cu8 \
    -t raw -r 1024000 -e unsigned-integer -b 8  -c 2 -L 1.024MSps.cu8 rate

# csdr
cat 2MSps.cf32 | csdr fractional_decimator_cc 1.953125 > 1.024MSps.cf32

# or capture at the target rate in the first place (best when you can re-capture)
rtl_sdr -s 1024000 -f 413016225 capture.cu8
```

2 MSPS → 1.024 MSPS is a clean 64/125 ratio. To check the effect of rate on a
given file, compare `--classify` with and without `--no-resample`.
