<div align="center">

# 🌌 Maya4

### Multi-Level SAR Processing & PyTorch DataLoader

*Unveiling the layers of Synthetic Aperture Radar data from Sentinel-1 missions*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![HF Organization](https://img.shields.io/badge/🤗%20Hugging%20Face-Maya4-yellow)](https://huggingface.co/Maya4)

[Overview](#-overview) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Processing Levels](#-processing-levels) •
[Citation](#-citation)

</div>

---

## 🎯 Overview

Maya4 is a production-ready Python package and dataset organization dedicated to curating and providing **multi-level intermediate SAR representations** from Sentinel-1 acquisitions, spanning the entire processing chain from Level 0 (raw) to Level 1 (focused imagery).

### The Māyā Philosophy

The name **Maya4** draws inspiration from the *Māyā veil* in philosophy, where reality is hidden behind successive layers—just as radar echoes undergo multiple transformations before forming a final SAR image. Each processing level reveals a different aspect of the electromagnetic interaction with Earth's surface.

### Why Maya4?

- **🎚️ Multi-Level Access**: Complete processing chain from raw echoes to focused imagery
- **🚀 Performance**: Zarr-based storage with intelligent chunk caching and lazy loading
- **🔧 Flexibility**: Access any intermediate representation for research and experimentation
- **☁️ Cloud-Native**: Native Hugging Face Hub integration with 68TB+ of curated data
- **📊 ML-Ready**: PyTorch-compatible dataloaders optimized for pre-training workflows
- **🌍 Geographic-Aware**: Built-in support for location-based clustering and filtering

---

## 🌐 Processing Levels

Maya4 exposes the complete SAR processing chain through intermediate signal representations:

| Level | Abbrev. | Description | Purpose / Value |
|-------|---------|-------------|-----------------|
| 📡 **Raw** | `raw` | Unprocessed radar echoes as recorded by Sentinel-1 | Baseline data; enables full custom SAR processing |
| 🎚️ **Range Compressed** | `rc` | Echoes compressed in range via matched filtering | Improved SNR; isolates scatterers along range |
| 🎯 **Range Cell Migration Corrected** | `rcmc` | Motion-compensated with corrected range migration | Preserves geometric fidelity; enables azimuth focusing |
| 🖼️ **Azimuth Compressed** | `ac` | Fully focused SAR image in slant-range geometry | Standard Level-1 product; interpretable imagery |

Each level represents a distinct transformation in the SAR focusing pipeline, allowing researchers to:
- **Experiment** with custom processing algorithms
- **Pre-train** deep learning models on intermediate representations
- **Analyze** signal characteristics at different processing stages
- **Develop** novel focusing techniques

---

## 📦 Pre-Training Datasets

Maya4 provides curated Pre-Training (PT) datasets in cloud-native Zarr format:

| Dataset Split | Contents | Acquisition Mode | Size | Hub Link |
|---------------|----------|------------------|------|----------|
| **PT1** | Multi-level SAR data | Stripmap | 17 TB | [🤗 Maya4/PT1](https://huggingface.co/datasets/Maya4/PT1) |
| **PT2** | Multi-level SAR data | Stripmap | 17 TB | [🤗 Maya4/PT2](https://huggingface.co/datasets/Maya4/PT2) |
| **PT3** | Multi-level SAR data | Stripmap | 17 TB | Coming Soon |
| **PT4** | Multi-level SAR data | Stripmap | 17 TB | [🤗 Maya4/PT4](https://huggingface.co/datasets/Maya4/PT4) |
| **Total** | — | — | **68 TB** | — |

*Data provided by the Copernicus Sentinel-1 mission (ESA)*

---

## ✨ Features

<table>
<tr>
<td width="50%">

### Core Capabilities
- **Multi-Level Data Access**  
  Complete processing chain from raw to focused
  
- **Zarr Backend**  
  Scalable, chunked storage for 68TB+ datasets
  
- **Normalization Suite**  
  MinMax, Z-Score, Robust, and Adaptive strategies
  
- **HuggingFace Integration**  
  Direct loading from Maya4 Hub repositories

</td>
<td width="50%">

### Advanced Features
- **Geographic Clustering**  
  Balanced sampling by location distribution
  
- **Positional Encoding**  
  Built-in transformer-compatible embeddings
  
- **Flexible Patch Modes**  
  Rectangular and parabolic extraction
  
- **Lazy Loading**  
  Memory-efficient processing of massive datasets

</td>
</tr>
</table>

---

## 📦 Installation

### Quick Install

```bash
# Using PDM (recommended)
pdm install

# Using pip
pip install -e .
```

### Environment-Specific Installation

<details>
<summary><b>Jupyter Environment</b></summary>

```bash
pdm install -G jupyter_env
```

Includes Jupyter notebook and lab dependencies for interactive development.
</details>

<details>
<summary><b>Geospatial Features</b></summary>

```bash
pdm install -G geospatial
```

Adds geographic processing tools and coordinate system support.
</details>

<details>
<summary><b>Development Setup</b></summary>

```bash
pdm install -G dev
```

Installs testing, linting, and development utilities.
</details>

<details>
<summary><b>Complete Installation</b></summary>

```bash
pdm install -G :all
```

Installs all optional dependencies for full functionality.
</details>

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

---

## 🚀 Quick Start

### Loading Maya4 Datasets

```python
from maya4 import get_sar_dataloader, SARTransform

# Configure normalization for specific processing levels
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=-3000,      # Range compressed min
    rc_max=3000,       # Range compressed max
    gt_min=-12000,     # Azimuth compressed (ground truth) min
    gt_max=12000,      # Azimuth compressed (ground truth) max
    complex_valued=True
)

# Load from Hugging Face Hub
dataloader = get_sar_dataloader(
    data_dir='hf://datasets/Maya4/PT1',  # Hugging Face dataset
    level_from='rcmc',                    # Input: Range Cell Migration Corrected
    level_to='ac',                        # Target: Azimuth Compressed (focused)
    batch_size=16,
    num_workers=4,
    patch_size=(1000, 1),
    stride=(300, 1),
    transform=transform,
    online=True,
    max_products=10
)

# Training loop across processing levels
for x_batch, y_batch in dataloader:
    # x_batch: RCMC intermediate representation
    # y_batch: Fully focused SAR image
    print(f'RCMC Input: {x_batch.shape} → Focused Output: {y_batch.shape}')
```

### Multi-Level Processing Example

```python
from maya4 import get_sar_dataloader, SampleFilter

# Filter for specific acquisition parameters
filters = SampleFilter(
    years=[2023],
    polarizations=['hh'],
    stripmap_modes=[1, 2, 3],
    parts=['PT1', 'PT3']
)

# Experiment with different processing level combinations
dataloader_raw_to_rc = get_sar_dataloader(
    data_dir='hf://datasets/Maya4/PT2',
    filters=filters,
    level_from='raw',      # Start from raw echoes
    level_to='rc',         # Learn range compression
    batch_size=16,
    patch_size=(1000, 100),
    buffer=(1000, 1000),
    stride=(300, 100),
    concatenate_patches=True,
    concat_axis=0,
    positional_encoding=True,
    cache_size=1000,
    verbose=True
)
```

### Production Configuration

```python
from maya4 import create_dataloaders

config = {
    'data_dir': '/Data/sar_focusing',
    'level_from': 'rcmc',
    'level_to': 'ac',
    'patch_size': [1000, 1],
    'buffer': [1000, 1000],
    'stride': [300, 1],
    'batch_size': 16,
    'num_workers': 4,
    'cache_size': 1000,
    'online': True,
    'concatenate_patches': True,
    'concat_axis': 0,
    'positional_encoding': True,
    'transforms': {
        'normalize': True,
        'normalization_type': 'minmax',
        'complex_valued': True,
        'rc_min': -3000,
        'rc_max': 3000,
        'gt_min': -12000,
        'gt_max': 12000
    },
    'train': {
        'batch_size': 16,
        'samples_per_prod': 1000,
        'max_products': 10,
        'filters': {
            'years': [2023],
            'polarizations': ['hh']
        }
    }
}

train_loader, val_loader, test_loader = create_dataloaders(config)
```

---

## 📚 Documentation

### Architecture Overview

```
maya4/
├── __init__.py              # Public API and package initialization
├── dataloader.py            # Core dataloader implementation
├── normalization.py         # Transformation and normalization modules
├── api.py                   # Hugging Face Hub integration
├── utils.py                 # Utility functions and helpers
├── location_utils.py        # Geographic processing utilities
└── dataset_creation.py      # Zarr dataset creation tools
```

### Core Components

#### 🗂️ SARZarrDataset

High-performance PyTorch Dataset for multi-level SAR data stored in Zarr format.

**Key Features:**
- Multi-mode patch sampling (rectangular, parabolic)
- LRU cache at chunk level for optimal performance
- Local and remote (HuggingFace Maya4) Zarr store support
- Advanced filtering: part, year, month, polarization, stripmap mode
- Support for all processing levels: raw, rc, rcmc, ac
- Positional encoding for transformer architectures
- Automatic patch concatenation for sequence models

**Example:**
```python
from maya4 import SARZarrDataset

# Access intermediate representations
dataset = SARZarrDataset(
    data_dir='hf://datasets/Maya4/PT1',
    level_from='rcmc',     # Input processing level
    level_to='ac',         # Target processing level
    patch_size=(1000, 100),
    stride=(300, 100),
    cache_size=500
)
```

#### 📊 SARDataloader

Custom DataLoader with intelligent KPatchSampler for balanced multi-product sampling.

**Key Features:**
- Configurable samples per product
- Multi-level shuffling (file and patch)
- Flexible patch ordering (row-major, column-major, chunk-based)
- Geographic clustering for distribution balance
- Automatic worker management

**Example:**
```python
from maya4 import SARDataloader

loader = SARDataloader(
    dataset=dataset,
    batch_size=32,
    samples_per_prod=500,
    shuffle=True,
    num_workers=8
)
```

#### 🔄 SARTransform

Modular transformation pipeline with multiple normalization strategies optimized for different processing levels.

**Supported Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **MinMax** | Scale to [0, 1] range | Standard neural network input across all levels |
| **Z-Score** | Mean/std standardization | Statistical normalization for intermediate representations |
| **Robust** | Median/IQR based | Outlier-resistant normalization for raw echoes |
| **Adaptive** | On-the-fly statistics | Dynamic normalization across processing levels |

**Example:**
```python
from maya4 import SARTransform

# MinMax normalization
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=-3000,
    rc_max=3000,
    complex_valued=True
)

# Robust normalization
transform = SARTransform.create_robust_normalized_transform(
    normalize=True,
    percentile_range=(5, 95)
)
```

#### 🌐 API Utilities

Seamless integration with Hugging Face Hub for accessing Maya4's 68TB+ datasets.

**Available Functions:**
- `list_base_files_in_repo()` - Repository file listing
- `fetch_chunk_from_hf_zarr()` - Selective chunk download
- `download_metadata_from_product()` - Metadata extraction

**Example:**
```python
from maya4.api import list_base_files_in_repo, fetch_chunk_from_hf_zarr

# List Maya4 PT1 dataset contents
files = list_base_files_in_repo('Maya4/PT1')

# Download specific chunk from processing level
chunk = fetch_chunk_from_hf_zarr(
    repo_id='Maya4/PT2',
    product_name='S1A_IW_SLC__1SDV',
    chunk_key='0.0',
    level='rcmc'
)
```

---

## 🎯 Use Cases

Maya4 enables a wide range of research and applications:

- **🧠 Deep Learning Pre-Training**: Train models on intermediate SAR representations
- **🔬 Custom Processing Algorithms**: Develop novel SAR focusing techniques
- **📊 Signal Analysis**: Study electromagnetic interactions at different processing stages
- **🎓 Education**: Understand the complete SAR processing pipeline
- **🚀 Model Development**: Build processing-level-aware neural networks
- **🌍 Large-Scale Research**: Access 68TB+ of curated Sentinel-1 data

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Contributors

**Roberto Del Prete**  
📧 roberto.delprete@esa.int  
🏢 European Space Agency

**Maya4 Organization**  
🤗 [Hugging Face Organization](https://huggingface.co/Maya4)  
🛰️ Data: Copernicus Sentinel-1 mission (ESA)

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📖 Citation

If you use Maya4 datasets or tools in your research, please cite:

```bibtex
@software{maya4_2024,
  author       = {Del Prete, Roberto and Maya4 Organization},
  title        = {Maya4: Multi-Level SAR Processing and Intermediate Representations},
  year         = {2024},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Maya4}},
  note         = {68TB+ curated Sentinel-1 Stripmap data spanning processing levels from raw to focused imagery}
}
```

---

## 📝 Changelog

### [0.1.0] - 2024-11-26

#### Added
- ✨ Initial release with multi-level SAR dataloader functionality
- 📡 Support for all processing levels: raw, rc, rcmc, ac
- 🔧 Multiple normalization strategies optimized for different levels
- ☁️ Hugging Face Hub integration for Maya4 PT datasets (68TB+)
- 🌍 Geographic clustering support for balanced sampling
- 📍 Positional encoding for transformer models
- ⚡ Lazy loading and intelligent chunk caching
- 📦 Zarr backend for cloud-native access

#### Features
- PyTorch-compatible DataLoader with custom samplers
- Flexible patch extraction modes
- Advanced filtering capabilities across acquisition parameters
- Production-ready configuration system for multi-level processing

---

## 🔗 Links

- **Hugging Face Organization**: [Maya4](https://huggingface.co/Maya4)
- **PT1 Dataset**: [Maya4/PT1](https://huggingface.co/datasets/Maya4/PT1) (17TB)
- **PT2 Dataset**: [Maya4/PT2](https://huggingface.co/datasets/Maya4/PT2) (17TB)
- **PT4 Dataset**: [Maya4/PT4](https://huggingface.co/datasets/Maya4/PT4) (17TB)
- **Issue Tracker**: [GitHub Issues](https://github.com/sirbastiano/maya4/issues)
- **Source Code**: [GitHub Repository](https://github.com/sirbastiano/maya4)

---

<div align="center">

**🌌 Unveiling the layers of SAR data, one transformation at a time**

Made with ❤️ by the Maya4 Team

**[⬆ back to top](#-maya4)**

</div>
