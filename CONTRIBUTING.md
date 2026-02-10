# Contributing to RTL-ML

First off, thanks for considering contributing to RTL-ML! 🎉

## Ways to Contribute

### 1. Add New Signal Types
The most valuable contribution! See [docs/ADDING_SIGNALS.md](docs/ADDING_SIGNALS.md)

Ideas:
- DMR/P25 digital voice
- LoRa IoT signals
- Weather fax (HF)
- SSB voice modes
- Satellite downlinks
- Your local signals!

### 2. Improve Feature Engineering
Enhance the 18-feature extractor in `src/signal_features.py`

Ideas:
- Wavelet features for burst detection
- Cyclostationary analysis
- Constellation diagram metrics
- Modulation-specific features

### 3. Port to New Hardware
Test and document on:
- Raspberry Pi Zero 2W
- Rock Pi
- LattePanda
- x86 machines
- Jetson Nano

### 4. Build Tools & Interfaces
- Web dashboard for monitoring
- Mobile app (termux + Python)
- Real-time waterfall classifier
- Batch scanning automation

### 5. Improve Documentation
- Video tutorials
- More examples
- Better troubleshooting
- Translations

### 6. Fix Bugs
Check the [Issues](https://github.com/TrevTron/rtl-ml/issues) tab for known bugs.

---

## Pull Request Process

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/rtl-ml.git
   cd rtl-ml
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/add-lora-signal
   # or
   git checkout -b fix/memory-leak
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update README if adding features

4. **Test thoroughly**
   ```bash
   # Test capture (requires RTL-SDR hardware)
   python src/capture_validated.py

   # Test training
   python src/train_validated.py

   # Test classification
   python src/classify_live.py --freq 98.7e6
   ```

5. **Commit with clear messages**
   ```bash
   git add .
   git commit -m "Add LoRa 868 MHz signal support"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/add-lora-signal
   # Then create PR on GitHub
   ```

7. **PR Description Template**
   ```markdown
   ## What does this PR do?
   Adds support for LoRa 868 MHz signals with chirp detection

   ## Changes made:
   - Added lora_868 to capture script
   - Captured 50 samples with validation
   - Achieved 89% accuracy after retraining
   - Included spectrogram in visualizations/

   ## Testing:
   - [x] Capture works
   - [x] Training improves accuracy
   - [x] Live classification correct
   - [x] Spectrogram looks good

   ## Screenshots:
   ![LoRa Spectrogram](visualizations/lora_868_spectrum.png)
   ```

---

## Code Style Guidelines

### Python (PEP 8)
```python
# Good
def extract_features(self, samples):
    """Extract 18 features from IQ samples."""
    power = np.abs(samples) ** 2
    return features

# Bad
def ExtractFeatures(self,samples):
    power=np.abs(samples)**2  # Missing docstring, spacing
    return features
```

### Naming Conventions
- **Functions**: `snake_case` - `extract_features()`
- **Classes**: `PascalCase` - `SignalFeatureExtractor`
- **Constants**: `UPPER_SNAKE` - `SAMPLE_RATE`
- **Variables**: `snake_case` - `fft_power`

### Documentation
- **Docstrings** for all public functions
- **Comments** for complex algorithms
- **Type hints** encouraged:
  ```python
  def classify_signal(frequency: float) -> tuple[str, float]:
      """Classify signal at given frequency."""
      pass
  ```

---

## Testing Requirements

### Before Submitting PR:

1. **Capture test** (if modifying capture):
   ```bash
   python src/capture_validated.py
   # Verify 30 samples per class saved
   ```

2. **Training test** (if modifying features/training):
   ```bash
   python src/train_validated.py
   # Accuracy should be ≥ 80%
   ```

3. **Classification test**:
   ```bash
   python src/classify_live.py --freq 98.7e6
   # Should classify correctly
   ```

4. **Spectrogram test**:
   ```bash
   # Check visualizations/*.png look correct
   # No blank images or errors
   ```

---

## Adding Dependencies

If your PR adds new Python packages:

1. Add to `requirements.txt`:
   ```
   your-new-package>=1.0.0
   ```

2. Explain why in PR description
3. Keep dependencies minimal (avoid bloat)

---

## Documentation Updates

If your PR changes functionality:

1. **Update README.md** - Add new signal to table, update examples
2. **Update docs/** - Modify relevant guides
3. **Add screenshots** - Show new features working
4. **Update CHANGELOG** - Add entry (if exists)

---

## Community Guidelines

### Be Respectful
- Constructive feedback only
- No harassment or discrimination
- Help newcomers learn

### Be Patient
- Reviews may take time
- Maintainers are volunteers
- Not all PRs can be merged

### Be Clear
- Explain your changes
- Provide context
- Link related issues

---

## Getting Help

### Questions about contributing?
- Open a [Discussion](https://github.com/TrevTron/rtl-ml/discussions)
- Ask in [Issues](https://github.com/TrevTron/rtl-ml/issues)
- Tag @TrevTron on Reddit (u/TrevTron)

### Found a bug?
1. Check [existing issues](https://github.com/TrevTron/rtl-ml/issues)
2. If new, open issue with:
   - Description
   - Steps to reproduce
   - Expected vs actual behavior
   - System info (OS, Python version, hardware)

---

## License

By contributing, you agree your code will be licensed under MIT License (same as the project).

You retain copyright but grant project rights to use/distribute your contribution.

---

## Recognition

Contributors are credited in:
- README.md Contributors section
- GitHub contributors graph
- Release notes (for significant contributions)

Thank you for making RTL-ML better! 🙏
