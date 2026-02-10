# Hardware Setup Guide

## Tested Configuration

### Indiedroid Nova ($179.95)
- **Purchase**: [AmeriDroid](https://ameridroid.com/products/indiedroid-nova?ref=ioqothsk)
- **Specs**: RK3588S, 16GB RAM, 6 TOPS NPU
- **OS**: Debian 12 (bookworm) - pre-installed
- **Connection**: SSH over local network

### RTL-SDR Blog V4 ($39.95)
- **Purchase**: [RTL-SDR.com](https://www.rtl-sdr.com/buy-rtl-sdr-dvb-t-dongles/)
- **Specs**: Rafael Micro R828D tuner, 24-1766 MHz, 0.5 PPM TCXO
- **Driver**: Built into Linux kernel (no install needed)
- **USB**: Use USB 2.0 or 3.0 port

**Total Cost: $219.90**

---

## Setup Steps

### 1. SSH into Nova
```bash
ssh debian@<nova-ip-address>
# Default password: 1234 (change this!)
```

### 2. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv librtlsdr-dev git
```

### 3. Create Project Directory
```bash
mkdir ~/rtl-ml && cd ~/rtl-ml
python3 -m venv venv
source venv/bin/activate
```

### 4. Clone Repository
```bash
git clone https://github.com/TrevTron/rtl-ml.git
cd rtl-ml
```

### 5. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 6. Test RTL-SDR
```bash
rtl_test -t
# Should show: "Found 1 device(s)"
# Press Ctrl+C to stop
```

### 7. Run Quick Start
```bash
python examples/quick_start.py
```

---

## Alternative Hardware

### Raspberry Pi 4/5
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- Same setup steps as Nova
- Use Raspberry Pi OS (64-bit)

### Orange Pi 5
- Requires Armbian OS
- Same setup otherwise
- Good performance/price ratio

### x86 Linux Desktop
- Any machine with USB ports
- Ubuntu/Debian recommended
- 8GB+ RAM for training

---

## Antenna Recommendations

### Included with RTL-SDR V4:
- 2x telescopic dipoles
- Good for FM/VHF/UHF
- Mount vertically for best results

### Upgrade Options:
- **ADS-B (1090 MHz)**: Dedicated 1090 MHz antenna ($20-40)
- **NOAA Satellites**: V-dipole or QFH antenna ($30-60)
- **Wideband**: Discone antenna (25-1300 MHz, $50-100)
- **Budget**: Simple wire dipole (free!)

---

## Network Configuration

### Finding Nova IP Address:
```bash
# On Nova:
hostname -I

# From Windows/Mac:
arp -a | findstr "b8-27"  # Look for Rockchip MAC prefix
```

### Static IP (Optional):
Edit `/etc/dhcpcd.conf`:
```
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

---

## Storage Recommendations

### Minimum:
- 32GB SD card for OS + captured data
- Full dataset: 2GB

### Recommended:
- 64GB+ for multiple datasets
- Fast SD card (UHS-I Class 10 minimum)

---

## Power Requirements

### Indiedroid Nova:
- 5V 3A USB-C power supply (included)
- RTL-SDR draws power from USB (no external needed)

### Raspberry Pi:
- Official 5V 3A power supply recommended
- Underpowered supplies cause stability issues

---

## Troubleshooting Hardware

### RTL-SDR not detected:
```bash
lsusb | grep RTL
# Should show: "Realtek Semiconductor Corp. RTL2838"
```

### Permission denied:
```bash
sudo usermod -a -G plugdev $USER
# Logout and log back in
```

### USB bandwidth issues:
Use ARM-optimized sample rate:
```python
sdr.sample_rate = 1.024e6  # Not 2.4e6!
```

---

## Next Steps

After hardware setup:
1. Capture your first dataset: `python src/capture_validated.py`
2. Train your model: `python src/train_validated.py`
3. Classify live signals: `python src/classify_live.py --freq 98.7e6`

See main README.md for full tutorial.
