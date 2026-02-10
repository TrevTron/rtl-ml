# Adding New Signal Types

This guide shows you how to extend RTL-ML with your own custom signal types.

---

## Step 1: Identify Your Target Signal

Research your signal before capturing:

### Find the Frequency
- Use [RadioReference.com](https://www.radioreference.com/)
- Check [SigIDWiki](https://www.sigidwiki.com/)
- Scan with `rtl_power` or GQRX

### Determine Modulation
- **FM**: Wideband FM broadcast, NBFM (voice)
- **AM**: Aviation, AM radio
- **Digital**: P25, DMR, LoRa, etc.
- **Packet**: AX.25, APRS, Packet radio

### Check Signal Activity
- Is it continuous (FM radio) or burst (pager)?
- Transmission schedule (satellite passes)?
- Time of day (more activity at certain hours)?

---

## Step 2: Modify `capture_validated.py`

Add your signal to the `signals` list:

```python
signals = [
    # ... existing signals ...
    
    # Your new signal
    {'label': 'FRS_radio', 'freq': 462.5625e6, 'desc': '📻 FRS Radio (462.56 MHz)'},
]
```

**Label naming convention:**
- Use `snake_case` (lowercase with underscores)
- Be descriptive: `weather_fax` not `wx`
- Avoid spaces or special characters

---

## Step 3: Capture Dataset

```bash
cd ~/rtl-ml
source venv/bin/activate
python src/capture_validated.py
```

The script will:
1. Tune to your frequency
2. Capture 30 samples (0.5 seconds each)
3. Generate a spectrogram PNG
4. Save samples to `datasets_validated/FRS_radio/`

**Important**: Ensure the signal is actively transmitting during capture!

---

## Step 4: Verify with Spectrogram

Check `visualizations/FRS_radio_spectrum.png`:

### Good Spectrogram:
- Clear signal visible
- Distinct from noise floor
- Consistent structure across samples

### Bad Spectrogram:
- All noise (no signal present)
- Extremely weak signal
- Same as noise baseline

If bad, recapture with:
- Better antenna placement
- Higher SDR gain (`sdr.gain = 49.6`)
- During known active times

---

## Step 5: Retrain Model

```bash
python src/train_validated.py
```

Watch the output:
```
Loading 9 classes: [..., 'FRS_radio']
  FRS_radio: 30 samples

Random Forest
   Cross-val: 0.850 (±0.030)
   Test acc: 0.88
```

Your new signal should achieve >70% accuracy. If lower:
- Capture more samples (50+)
- Check for interference
- Verify signal characteristics are unique

---

## Step 6: Test Classification

```bash
python src/classify_live.py --freq 462.5625e6
```

Expected output:
```
Signal: FRS_radio
Confidence: 89.2%
```

---

## Example: Adding LoRa (868 MHz)

### 1. Add to capture script:
```python
{'label': 'lora_868', 'freq': 868e6, 'desc': '📡 LoRa 868 MHz'},
```

### 2. Capture during active transmissions:
```bash
# Have a LoRa device transmitting nearby
python src/capture_validated.py
```

### 3. Check features differentiate it:
```python
# LoRa has unique chirp patterns
# Features that help:
# - Bandwidth ratio (spread spectrum)
# - Phase characteristics (chirps)
# - FFT shape (sweeping frequency)
```

### 4. Retrain and test:
```bash
python src/train_validated.py
python src/classify_live.py --freq 868e6
```

---

## Tips for Good Signal Separation

### Choose Signals with Distinct Characteristics:

**Good separation:**
- ADS-B (bursts) vs FM (continuous)
- Narrowband (pager) vs Wideband (FM)
- Packet (APRS) vs Voice (NFM)

**Poor separation:**
- Two NFM voice stations (same modulation)
- Two LoRa devices (same parameters)
- Similar digital modes (P25/DMR might confuse)

### Increase Feature Discrimination:

If your signal confuses with existing ones, modify `signal_features.py`:

```python
# Add signal-specific features
# Example: Detect chirps for LoRa
chirp_rate = detect_chirp_rate(samples)
features.append(chirp_rate)
```

---

## Validation Functions

Add custom validation in `capture_validated.py`:

```python
def validate_lora(self, samples):
    """Validate LoRa by detecting chirps"""
    # Your chirp detection logic
    has_chirps = detect_chirps(samples)
    
    return {
        'has_chirps': bool(has_chirps),
        'chirp_rate': float(chirp_rate),
        'note': 'LoRa uses frequency-swept chirps'
    }
```

Then route to it:
```python
elif label == 'lora_868':
    return self.validate_lora(samples)
```

---

## Contributing Your Signal

Share your new signal type with the community:

1. **Test thoroughly** (>80% accuracy)
2. **Document** in README what it is
3. **Include spectrogram** in PR
4. **Explain use case** (why it's useful)

Submit PR to: [github.com/TrevTron/rtl-ml](https://github.com/TrevTron/rtl-ml)

---

## Advanced: Adding Decoder Validation

For signals with existing decoders:

### Example: FRS Radio (NFM voice)
```bash
# Use rtl_fm to decode
rtl_fm -f 462.5625M -M fm | aplay

# Measure audio SNR
rtl_fm -f 462.5625M -M fm | sox -t raw -r 24k -e s -b 16 -c 1 - -n stat
```

### Example: AIS (Marine tracking)
```bash
# Use rtl_ais decoder
rtl_ais -n

# Check for valid NMEA sentences
```

Add validation result to JSON for proof.

---

## Troubleshooting New Signals

### Signal not captured properly:
- Check frequency is correct (use GQRX to verify)
- Ensure antenna covers that frequency
- Verify signal is actually transmitting

### Low accuracy after training:
- Capture more samples (50-100)
- Check spectrogram looks distinct from noise
- Add signal-specific features

### Confusion with existing signals:
- Check which signal it confuses with (confusion matrix)
- Add discriminative features
- Ensure labels are correct (maybe they're actually similar?)

---

## Next Steps

After successfully adding a signal:
1. Update README with new signal in table
2. Add to examples/batch_classify.py
3. Document any special requirements
4. Share results on r/RTLSDR!
