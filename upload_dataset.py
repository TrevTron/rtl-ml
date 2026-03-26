#!/usr/bin/env python3
"""
Upload RTL-ML Dataset v2 to Hugging Face
Uploads the validated 800-sample dataset (7 classes)
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

def create_dataset_readme():
    """Create README.md for the dataset"""
    return """# RTL-ML Dataset v2

## Dataset Summary

This dataset contains 800 validated RF signal samples captured using an RTL-SDR Blog V4 dongle on an Indiedroid Nova (RK3588S). Designed for training machine learning models to classify common RF signals.

**Samples:** 800 (7 classes)  
**Format:** NumPy arrays (.npy files) — each file is a dict with IQ data + metadata  
**Sample Rate:** 1.024 MSPS  
**Sample Duration:** 0.5 seconds per capture  
**Quality Gates:** DC removal, auto-gain, 6 dB minimum SNR, per-class validation  

## Signal Classes

| Class | Frequency | Count | Description |
|-------|-----------|-------|-------------|
| FM_broadcast | 88.5, 93.3, 98.7, 101.1, 105.7 MHz | 200 | Commercial FM radio (5 stations) |
| NOAA_weather | 162.4 MHz | 100 | Weather radio broadcasts |
| APRS | 144.39 MHz | 100 | Amateur radio position reporting |
| pager | 152.84 MHz | 100 | POCSAG pager transmissions |
| ISM_sensors | 433.92 MHz | 100 | Wireless sensors & remote controls |
| FRS_GMRS | 462.5625 MHz | 100 | Family/general mobile radio |
| noise | 145.0 MHz | 100 | Background RF noise baseline |

## What Changed from v1

- **7 classes** (removed ADS-B — 1090 MHz out of R828D tuner range; removed NOAA APT — decommissioned Aug 2025; added FRS/GMRS)
- **800 samples** (up from 240) with 100+ per class
- **DC offset removal** on every capture (`samples -= np.mean(samples)`)
- **Auto-gain calibration** per frequency
- **6 dB SNR gate** — rejects weak/empty captures
- **Per-class quality validators** (bandwidth, burst ratio, packet detection)
- **Temporal train/test split** — first 80% train, last 20% test (no data leakage)
- **Multi-frequency FM** — trained on 5 stations for frequency-invariant classification
- **Metadata in every file** — center_freq, sample_rate, timestamp, label, snr_db, version

## Model Performance

- **Random Forest:** 96.9% accuracy (155/160 test samples correct)
- **Temporal split:** No data leakage between train and test
- **Cross-frequency FM:** Generalizes to unseen FM stations

## Sample Format

Each .npy file contains a dict:
```python
{
    'samples': np.array([...], dtype=complex64),  # IQ data
    'center_freq': 98700000.0,
    'sample_rate': 1024000.0,
    'timestamp': '2026-01-15T14:23:01',
    'label': 'FM_broadcast',
    'duration': 0.5,
    'snr_db': 17.5,
    'version': 'v2'
}
```

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
data = np.load(f"{dataset_path}/datasets_validated/FM_broadcast/FM_broadcast_0.npy", allow_pickle=True).item()
print(f"Signal: {data['label']}, SNR: {data['snr_db']:.1f} dB, Freq: {data['center_freq']/1e6:.1f} MHz")
```

## Hardware

- **SDR:** RTL-SDR Blog V4 ($39.95) — **requires [RTL-SDR Blog driver fork](https://github.com/rtlsdrblog/rtl-sdr-blog)** for R828D tuner support
- **Computer:** Indiedroid Nova 16GB ($179.95)
- **Antenna:** Telescopic dipole (included with V4)

## Citation

```bibtex
@misc{rtl-ml-dataset-v2,
  author = {TrevTron},
  title = {RTL-ML Dataset v2: Validated RF Signal Captures},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/TrevTron/rtl-ml-dataset}}
}
```

## License

MIT License - Free for commercial and non-commercial use.

## Related

- **Code:** [github.com/TrevTron/rtl-ml](https://github.com/TrevTron/rtl-ml)
"""

def upload_dataset(dataset_dir, repo_name="rtl-ml-dataset"):
    """Upload dataset to Hugging Face"""
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    # Count files (v2 uses subdirectories per class)
    npy_files = list(dataset_path.rglob("*.npy"))
    if len(npy_files) < 700:
        print(f"⚠️  Warning: Expected ~800 files, found {len(npy_files)}")
    
    print(f"Found {len(npy_files)} .npy files ({sum(f.stat().st_size for f in npy_files) / 1e9:.2f} GB)")
    
    try:
        api = HfApi()
        user = api.whoami()
        username = user['name']
        repo_id = f"{username}/{repo_name}"
        
        print(f"\n📦 Creating dataset repository: {repo_id}")
        
        # Create repository
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            print(f"✅ Repository created: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Note: {e}")
        
        # Upload README
        print("\n📄 Uploading README.md...")
        readme_content = create_dataset_readme()
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ README uploaded")
        
        # Upload all .npy files
        print(f"\n📤 Uploading {len(npy_files)} signal files...")
        print("This will take several minutes for 1.9 GB...")
        
        api.upload_folder(
            folder_path=str(dataset_path),
            path_in_repo="datasets_validated",
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="*.npy"
        )
        
        print(f"\n✅ Upload complete!")
        print(f"\n🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\n📥 Download with:")
        print(f'    snapshot_download(repo_id="{repo_id}", repo_type="dataset")')
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Dataset location
    dataset_dir = r"C:\Users\tre77\OneDrive\Desktop\pentesting\zzzzzzz\datasets_validated"
    
    print("=" * 60)
    print("RTL-ML Dataset Upload to Hugging Face")
    print("=" * 60)
    print(f"\nDataset directory: {dataset_dir}")
    
    success = upload_dataset(dataset_dir)
    
    if success:
        print("\n✅ All done! Dataset is live on Hugging Face.")
        sys.exit(0)
    else:
        print("\n❌ Upload failed. Check errors above.")
        sys.exit(1)
