# Troubleshooting Guide

## Common Issues & Solutions

---

### ⚠️ "PLL not locked" warnings

**Symptom:**
```
[R82XX] PLL not locked!
```

**Cause:** Normal RTL-SDR behavior during frequency tuning

**Impact:** None - captures work fine despite warnings

**Fix:** Ignore these warnings (they're cosmetic)

---

### ⚠️ USB overflow / lost samples

**Symptom:**
```
Dropped samples detected!
USB buffer overflow
```

**Cause:** Sample rate too high for ARM USB bandwidth

**Fix:** Use ARM-optimized rate in capture scripts:
```python
sdr.sample_rate = 1.024e6  # NOT 2.4e6
```

---

### ⚠️ "No RTL-SDR devices found"

**Symptom:**
```
usb.core.USBError: [Errno 13] Access denied
```

**Cause:** Permission issue or device not connected

**Fix:**
```bash
# Check device exists
lsusb | grep RTL

# Add udev rule
sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"0bda\", ATTRS{idProduct}==\"2838\", MODE=\"0666\"" > /etc/udev/rules.d/rtl-sdr.rules'

# Reload rules
sudo udevadm control --reload-rules

# Unplug and replug RTL-SDR
```

---

### ⚠️ Low classification accuracy (< 70%)

**Possible Causes:**
1. Wrong antenna for frequency range
2. Local interference/noise
3. Weak signal (antenna placement)
4. Different signal characteristics in your area

**Fixes:**

**1. Check antenna:**
```python
# FM needs VHF antenna (88-108 MHz)
# ADS-B needs 1090 MHz antenna
# Use appropriate antenna or wideband discone
```

**2. Verify signals are present:**
```bash
# Use rtl_power to scan spectrum
rtl_power -f 88M:108M:1M -i 1 scan.csv
```

**3. Capture more samples:**
```python
# In capture_validated.py, change:
samples_per_class = 50  # Was 30
```

**4. Retrain model:**
```bash
python src/train_validated.py
```

---

### ⚠️ "Model file not found"

**Symptom:**
```python
FileNotFoundError: models/rtl_classifier_validated.pkl
```

**Cause:** Pre-trained model missing or wrong path

**Fix:**
```bash
# Option A: Train your own model
python src/train_validated.py

# Option B: Check path in classify_live.py
python src/classify_live.py --model models/rtl_classifier_validated.pkl --freq 98.7e6
```

---

### ⚠️ ImportError: No module named 'rtlsdr'

**Symptom:**
```python
ModuleNotFoundError: No module named 'rtlsdr'
```

**Cause:** pyrtlsdr not installed or wrong virtual environment

**Fix:**
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### ⚠️ GLIBC version error

**Symptom:**
```
ImportError: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
```

**Cause:** Python packages built for newer GLIBC than system has

**Fix:**
```bash
# Update system
sudo apt update && sudo apt upgrade

# Or use older numpy
pip install numpy==1.24.0
```

---

### ⚠️ Memory error during training

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Cause:** Insufficient RAM (< 4GB) or too many samples

**Fix:**
```bash
# Check available RAM
free -h

# Reduce dataset size temporarily
# Or use smaller test split
```

In `train_validated.py`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Increased from 0.2
    random_state=42, stratify=y
)
```

---

### ⚠️ Signal validation fails

**Symptom:**
```
ISM sensors: 0.0x burst ratio (expected > 5x)
```

**Cause:** No active transmissions during capture window

**Fix:**
```python
# Increase capture duration in capture_validated.py
vis_samples = capture.capture_signal(freq, duration=2.0)  # Was 1.0

# Or capture at known active times
# (e.g., NOAA satellite must be overhead)
```

---

### ⚠️ Classification always predicts "noise"

**Cause:** Empty frequency or very weak signals

**Fix:**
1. Use `rtl_fm` to verify signal exists:
```bash
rtl_fm -f 98.7M -M wbfm | aplay
# Should hear FM radio
```

2. Increase SDR gain:
```python
sdr.gain = 49.6  # Maximum gain instead of 40
```

3. Check antenna connection (loose cable?)

---

### ⚠️ Spectrograms look blank/wrong

**Cause:** matplotlib backend issues or zeros in data

**Fix:**
```python
# In capture_validated.py, add debug:
print(f"Sample power: {np.mean(np.abs(samples)**2)}")
print(f"Sample range: {np.min(np.abs(samples))} - {np.max(np.abs(samples))}")

# Should see non-zero power
```

---

##⚠️ Git LFS bandwidth exceeded

**Symptom:**
```
This repository is over its data quota
```

**Cause:** Large dataset files (1.9GB) pushing to GitHub

**Fix:** Use Hugging Face Datasets instead:
```bash
# Upload dataset to Hugging Face
https://huggingface.co/datasets/TrevTron/rtl-ml-dataset
```

See main README for Hugging Face instructions.

---

## Still Having Issues?

1. **Check the logs**: Most errors print helpful messages
2. **Verify hardware**: `rtl_test -t` should pass
3. **Check GitHub Issues**: [github.com/TrevTron/rtl-ml/issues](https://github.com/TrevTron/rtl-ml/issues)
4. **Reddit help**: r/RTLSDR community is very helpful
5. **Open an issue**: Include error messages + system info

**System Info Template:**
```bash
echo "OS: $(uname -a)"
echo "Python: $(python --version)"
echo "RTL-SDR: $(rtl_test 2>&1 | head -5)"
pip list | grep -E "(rtlsdr|numpy|scikit)"
```
