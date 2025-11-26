<div align="center">

# 🛰️ Maya4

### Advanced SAR Data Processing & PyTorch DataLoader

*High-performance toolkit for Synthetic Aperture Radar (SAR) data from Sentinel-1 missions*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Citation](#-citation)

</div>

---

## 🎯 Overview

Maya4 is a production-ready Python package designed for efficient processing and loading of Synthetic Aperture Radar (SAR) data. Built with PyTorch integration and optimized for large-scale machine learning workflows, it provides a comprehensive suite of tools for SAR data manipulation, normalization, and batch processing.

### Why Maya4?

- **🚀 Performance**: Zarr-based storage with intelligent chunk caching and lazy loading
- **🔧 Flexibility**: Modular architecture supporting multiple normalization strategies
- **☁️ Cloud-Ready**: Native Hugging Face Hub integration for remote data access
- **📊 ML-Optimized**: PyTorch-compatible dataloaders with advanced sampling strategies
- **🌍 Geographic-Aware**: Built-in support for location-based clustering and filtering

---

## ✨ Features

<table>
<tr>
<td width="50%">

### Core Capabilities
- **Efficient SAR Dataloader**  
  PyTorch-compatible with patch-based sampling
  
- **Zarr Backend**  
  Fast, chunked storage for large datasets
  
- **Normalization Suite**  
  MinMax, Z-Score, Robust, and Adaptive strategies
  
- **HuggingFace Integration**  
  Direct loading from Hub repositories

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
  Memory-efficient coordinate generation

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

### Basic Example

```python
from maya4 import get_sar_dataloader, SARTransform

# Configure normalization
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=-3000,
    rc_max=3000,
    gt_min=-12000,
    gt_max=12000,
    complex_valued=True
)

# Initialize dataloader
dataloader = get_sar_dataloader(
    data_dir='/path/to/sar_data',
    level_from='rcmc',
    level_to='az',
    batch_size=16,
    num_workers=4,
    patch_size=(1000, 1),
    stride=(300, 1),
    transform=transform,
    online=True,
    max_products=10
)

# Training loop
for x_batch, y_batch in dataloader:
    # x_batch: Input SAR patches
    # y_batch: Target ground truth
    print(f'Input: {x_batch.shape} | Target: {y_batch.shape}')
```

### Advanced Configuration

```python
from maya4 import SampleFilter, get_sar_dataloader

# Define data filters
filters = SampleFilter(
    years=[2023],
    polarizations=['hh'],
    stripmap_modes=[1, 2, 3],
    parts=['PT1', 'PT3']
)

# Configure advanced dataloader
dataloader = get_sar_dataloader(
    data_dir='/path/to/data',
    filters=filters,
    level_from='rcmc',
    level_to='az',
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
    'level_to': 'az',
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

High-performance PyTorch Dataset for SAR data stored in Zarr format.

**Key Features:**
- Multi-mode patch sampling (rectangular, parabolic)
- LRU cache at chunk level for optimal performance
- Local and remote (HuggingFace) Zarr store support
- Advanced filtering: part, year, month, polarization, stripmap mode
- Positional encoding for transformer architectures
- Automatic patch concatenation for sequence models

**Example:**
```python
from maya4 import SARZarrDataset

dataset = SARZarrDataset(
    data_dir='/path/to/zarr',
    level_from='rcmc',
    level_to='az',
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

Modular transformation pipeline with multiple normalization strategies.

**Supported Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| **MinMax** | Scale to [0, 1] range | Standard neural network input |
| **Z-Score** | Mean/std standardization | Statistical normalization |
| **Robust** | Median/IQR based | Outlier-resistant normalization |
| **Adaptive** | On-the-fly statistics | Dynamic data distributions |

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

Seamless integration with Hugging Face Hub for cloud-based workflows.

**Available Functions:**
- `list_base_files_in_repo()` - Repository file listing
- `fetch_chunk_from_hf_zarr()` - Selective chunk download
- `download_metadata_from_product()` - Metadata extraction

**Example:**
```python
from maya4.api import list_base_files_in_repo, fetch_chunk_from_hf_zarr

# List repository contents
files = list_base_files_in_repo('username/sar-dataset')

# Download specific chunk
chunk = fetch_chunk_from_hf_zarr(
    repo_id='username/sar-dataset',
    product_name='S1A_IW_SLC__1SDV',
    chunk_key='0.0'
)
```

---

## 🔧 Dependencies

### Core Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.5.2 | Data manipulation |
| numpy | ≥ 1.24.0 | Numerical computing |
| torch | ≥ 2.0.0 | Deep learning framework |
| zarr | ≥ 2.14.0 | Chunked array storage |
| dask[array] | ≥ 2023.5.0 | Parallel computing |
| tqdm | ≥ 4.65.0 | Progress bars |
| matplotlib | ≥ 3.7.0 | Visualization |
| scikit-learn | ≥ 1.3.0 | ML utilities |
| huggingface-hub | ≥ 0.16.0 | Hub integration |

### Optional Dependencies

- **jupyter_env**: Interactive development (Jupyter Lab/Notebook)
- **geospatial**: Geographic processing (GeoPandas, Shapely)
- **dev**: Testing and development (pytest, black, mypy)
- **docs**: Documentation generation (Sphinx, mkdocs)

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Contributors

**Roberto Del Prete**  
📧 roberto.delprete@esa.int  
🏢 European Space Agency

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📖 Citation

If you use Maya4 in your research, please cite:

```bibtex
@software{maya4_2024,
  author       = {Del Prete, Roberto},
  title        = {Maya4: Advanced SAR Data Processing and DataLoader},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/sirbastiano/maya4},
  version      = {0.1.0}
}
```

---

## 📝 Changelog

### [0.1.0] - 2024-11-26

#### Added
- ✨ Initial release with core dataloader functionality
- 🔧 Multiple normalization strategies (MinMax, Z-Score, Robust, Adaptive)
- ☁️ Hugging Face Hub integration for remote data access
- 🌍 Geographic clustering support for balanced sampling
- 📍 Positional encoding for transformer models
- ⚡ Lazy loading and intelligent chunk caching
- 📦 Zarr backend for efficient storage and retrieval

#### Features
- PyTorch-compatible DataLoader with custom samplers
- Flexible patch extraction modes
- Advanced filtering capabilities
- Production-ready configuration system

---

## 🔗 Links

- **Documentation**: [Coming Soon]
- **Issue Tracker**: [GitHub Issues](https://github.com/sirbastiano/maya4/issues)
- **Source Code**: [GitHub Repository](https://github.com/sirbastiano/maya4)
- **Discussions**: [GitHub Discussions](https://github.com/sirbastiano/maya4/discussions)

---

<div align="center">

Made with ❤️ by the SAR Data Processing Team

**[⬆ back to top](#-maya4)**

</div>
