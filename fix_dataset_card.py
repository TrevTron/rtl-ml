#!/usr/bin/env python3
"""
Fix Hugging Face Dataset Card with proper YAML metadata
"""

from huggingface_hub import HfApi

def create_fixed_readme():
    """Create README.md with proper YAML frontmatter"""
    return """---
license: mit
task_categories:
- time-series-forecasting
- audio-classification
language:
- en
tags:
- rf-signals
- radio
- rtl-sdr
- signal-processing
- machine-learning
- telecommunications
- software-defined-radio
pretty_name: RTL-ML RF Signal Classification Dataset
size_categories:
- 100K<n<1M
---

# RTL-ML Dataset

> ⚠️ **v1.0 Dataset Notice**: Community feedback identified that several signal classes have suboptimal SNR due to antenna/gain configuration during capture. The classifier achieves stated accuracy but is partially learning frequency-specific noise characteristics rather than pure signal features. v2.0 with improved captures is in progress.

## Dataset Summary

This dataset contains 240 validated RF signal samples captured using an RTL-SDR Blog V4 dongle. It's designed for training machine learning models to classify common RF signals.

**Total Size:** 1.9 GB  
**Samples:** 240 (30 samples × 8 classes)  
**Format:** NumPy arrays (.npy files)  
**Sample Rate:** 1.024 MSPS  
**Sample Duration:** 1 second per capture  

## Signal Classes

| Class | Frequency | Count | Description |
|-------|-----------|-------|-------------|
| ADS_B | 1090 MHz | 30 | Aircraft transponder signals |
| APRS | 144.39 MHz | 30 | Amateur radio position reporting |
| FM_broadcast | 88-108 MHz | 30 | Commercial FM radio stations |
| ISM_sensors | 433.92 MHz | 30 | Wireless sensors & remote controls |
| NOAA_APT | 137.5 MHz | 30 | Weather satellite imagery |
| NOAA_weather | 162.4 MHz | 30 | Weather radio broadcasts |
| noise | Various | 30 | Background RF noise baseline |
| pager | 152.84 MHz | 30 | POCSAG pager transmissions |

## Validation Metrics

- **ISM Sensors:** 20.6x burst ratio (strong on/off keying)
- **NOAA Weather:** 14.4 dB SNR (clear signal)
- **Pager/APRS:** 12.7 dB SNR (good quality)
- **Model Accuracy:** 87.5% on test set

## Usage

```python
from huggingface_hub import snapshot_download
import numpy as np

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="TrevTron/rtl-ml-dataset",
    repo_type="dataset"
)

# Load a sample
sample = np.load(f"{dataset_path}/datasets_validated/ADS_B_0.npy")
print(f"Signal shape: {sample.shape}")  # (1048576,) complex64
```

## Dataset Structure

```
rtl-ml-dataset/
└── datasets_validated/
    ├── ADS_B_0.npy ... ADS_B_29.npy       (30 files)
    ├── APRS_0.npy ... APRS_29.npy         (30 files)
    ├── FM_broadcast_0.npy ... _29.npy     (30 files)
    ├── ISM_sensors_0.npy ... _29.npy      (30 files)
    ├── NOAA_APT_0.npy ... NOAA_APT_29.npy (30 files)
    ├── NOAA_weather_0.npy ... _29.npy     (30 files)
    ├── noise_0.npy ... noise_29.npy       (30 files)
    └── pager_0.npy ... pager_29.npy       (30 files)
```

Each `.npy` file contains:
- **Shape:** (1048576,) - 1 second @ 1.024 MSPS
- **Dtype:** `complex64` (I/Q samples)
- **Size:** ~8.4 MB per file

## Hardware

- **SDR:** RTL-SDR Blog V4 ($39.95)
- **Computer:** Indiedroid Nova 16GB or Raspberry Pi 4/5 (8GB+ recommended)
- **Antenna:** Telescopic dipole (included)

## Model Performance

When trained with Random Forest (100 trees):
- **Overall Accuracy:** 87.5%
- **Perfect Classes:** ADS-B, FM, ISM, NOAA APT, Weather, Pager (100%)
- **Challenging:** APRS ↔ Noise confusion (sparse packets)

## Citation

```bibtex
@misc{rtl-ml-dataset,
  author = {TrevTron},
  title = {RTL-ML Dataset: Validated RF Signal Captures},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/TrevTron/rtl-ml-dataset}}
}
```

## License

MIT License - Free for commercial and non-commercial use.

## Related

- **Code:** [github.com/TrevTron/rtl-ml](https://github.com/TrevTron/rtl-ml)
- **Blog:** [unland.dev/blog/building-ai-radio-scanner-rtl-sdr-machine-learning](https://unland.dev/blog/building-ai-radio-scanner-rtl-sdr-machine-learning)
- **Hardware Guide:** [Indiedroid Nova Setup](https://github.com/TrevTron/rtl-ml/blob/main/docs/HARDWARE_SETUP.md)

## Contributions

Captured in Temecula, CA (Southern California) using:
- Clear line of sight to multiple signal sources
- Validated with spectral analysis and manual inspection
- All samples meet minimum SNR requirements (>10 dB for modulated signals)

For questions or improvements, see the [GitHub repository](https://github.com/TrevTron/rtl-ml).
"""

def fix_dataset_card():
    """Upload fixed README to Hugging Face"""
    try:
        api = HfApi()
        repo_id = "TrevTron/rtl-ml-dataset"
        
        print("=" * 60)
        print("Fixing Hugging Face Dataset Card")
        print("=" * 60)
        print(f"\nRepository: {repo_id}")
        
        readme_content = create_fixed_readme()
        
        print("\n📄 Uploading fixed README.md with YAML metadata...")
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add YAML metadata to fix dataset card warnings"
        )
        
        print("✅ Dataset card fixed!")
        print(f"\n🔗 View at: https://huggingface.co/datasets/{repo_id}")
        print("\nChanges:")
        print("  ✅ Added YAML frontmatter with license, tags, categories")
        print("  ✅ Added dataset structure section")
        print("  ✅ Added model performance details")
        print("  ✅ Fixed metadata warnings")
        print("\nNote: Dataset viewer may take a few minutes to update.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_dataset_card()
