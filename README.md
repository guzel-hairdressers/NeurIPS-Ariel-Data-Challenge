
# Ariel Data Challenge 2025

> **A production-ready baseline pipeline for exoplanet atmospheric analysis using ESA's Ariel Space Mission simulated data**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Ariel%20Data%20Challenge%202025-20BEFF.svg)](https://www.kaggle.com/competitions/ariel-data-challenge-2025)

## Overview

This repository contains a complete **baseline solution** for the [Ariel Data Challenge 2025](https://www.kaggle.com/competitions/ariel-data-challenge-2025) - a cutting-edge competition focused on extracting atmospheric composition from exoplanet transit spectroscopy data.

The pipeline processes raw infrared detector data from the simulated **AIRS (Atmospheric Infrared Spectrometer)** instrument and produces high-quality transmission spectra that reveal the atmospheric properties of distant worlds.

## Key Features

### Advanced Calibration Pipeline
- **Correlated Double Sampling (CDS)**: Eliminates instrumental drift and low-frequency noise
- **Hot/Dead Pixel Masking**: Statistical detection using sigma-clipping with outlier rejection
- **Flat-field Correction**: Pixel-to-pixel sensitivity normalization
- **Intelligent Temporal Binning**: SNR optimization through adaptive frame averaging

### Robust Detrending System
- **Two-pass Detrending Architecture**: 
  - **Pass 1**: Linear trend removal with sigma-clipped masking
  - **Pass 2**: Polynomial refinement on pre-normalized data
- **KMeans-based Transit Detection**: Automatic separation of in/out-of-transit data
- **Spectral Channel Optimization**: Simplified robust processing for noisy wavelength bins

### Scientific Analysis Tools
- **Transmission Spectroscopy**: Wavelength-dependent transit depth extraction
- **Spectral Binning**: Professional mapping from 356 detector columns → 283 standardized bins
- **Statistical Error Analysis**: Plateau scatter-based uncertainty estimation
- **Publication-Quality Output**: Normalized light curves ready for scientific interpretation

## Pipeline Architecture

```
Raw AIRS Data → Calibration → CDS Processing → Flat-field → Temporal Binning
     ↓
Light Curve Extraction → Detrending → Normalization → Transit Depth Measurement
     ↓
Spectral Analysis → Wavelength Mapping → Transmission Spectrum → Competition CSV
```

## Quick Start

### Installation

# Clone the repository
```
git clone https://github.com/guzel-hairdressers/neurips-ariel-data-challenge.git
cd neurips-ariel-data-challenge
```
# Install dependencies
```
pip install -r requirements.txt
```

### Basic Usage

```
from src.ariel_pipeline import calibrate_airs_ch0_advanced, build_spectrum
```
# Process single planet
```
processed_signal, frames = calibrate_airs_ch0_advanced(
    signal_path='path/to/planet/AIRS-CH0_signal_0.parquet',
    calibration_path='path/to/planet/AIRS-CH0_calibration_0',
    bin_size=10
)
```

# Generate transmission spectrum
```
spectrum_data = build_spectrum(
    root_path='path/to/data/',
    max_planets=1
)
```

## Core Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `calibrate_airs_ch0_advanced()` | Complete calibration with CDS + binning | Raw data → Clean signal |
| `detrend_light_curve()` | Two-pass robust detrending | Trend removal + normalization |
| `analyze_light_curves()` | Full analysis with visualization | Quality control + diagnostics |
| `build_spectrum()` | Transmission spectrum extraction | Scientific analysis |
| `calc_submission()` | Batch processing pipeline | Competition submission |

## Performance & Results

### Output Quality
- **Clean normalized light curves** with typical transit depths: 0.1-2.0%
- **Transmission spectra** revealing atmospheric absorption features
- **Competition-ready CSV** in required format (283 wavelengths + 283 uncertainties)
- **Robust error handling** for corrupted/edge-case data

## Scientific Methodology

### Detector Physics
- Proper handling of infrared detector noise characteristics
- CDS implementation following astronomical best practices
- Flat-field correction accounting for pixel variations

### Statistical Rigor
- Sigma-clipping for outlier-resistant trend identification
- Bootstrap-ready error estimation framework
- Physically-motivated parameter selection

### Competition Strategy
This baseline demonstrates understanding of:
- Raw detector data processing
- Astronomical calibration standards
- Robust trend removal techniques
- Scientific error analysis
- Production-ready code architecture


## Technical Requirements

```
python>=3.8
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
astropy>=5.0.0
tqdm>=4.60.0
```

## Usage Examples

### Single Planet Analysis
```
# Analyze one planet with full visualization
analyze_light_curves(
    root_path='/path/to/train/',
    max_planets=1,
    bin_size=10
)
```

### Batch Processing
# Generate competition submission
```
calc_submission(
    root_path='/path/to/test/',
    list_planets=test_planet_ids,
    wavelengths_csv='wavelengths.csv',
    axis_info_parquet='axis_info.parquet',
    out_csv='baseline_submission.csv'
)
```

## Competition Performance

This baseline solution provides:
- **Solid foundation** for advanced ML approaches
- **Competitive scores** through physics-based processing
- **Robust handling** of real-world data challenges
- **Scalable architecture** for parameter optimization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Star the repository if you find it useful!

## Links

- [Ariel Data Challenge 2025](https://www.kaggle.com/competitions/ariel-data-challenge-2025)
- [ESA Ariel Mission](https://www.cosmos.esa.int/web/ariel)
- [Competition Discussion Forum](https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion)

---

**Built for the Ariel Data Challenge 2025** | *Extracting secrets from distant worlds, one photon at a time* 
```
