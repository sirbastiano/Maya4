# Maya4 - SAR Data Processing and Dataloader

A Python package for efficient processing and loading of Synthetic Aperture Radar (SAR) data from Sentinel-1 missions.

## Features

- **Efficient SAR Dataloader**: PyTorch-compatible dataloader with patch-based sampling
- **Zarr Backend**: Fast, chunked storage format for large SAR datasets
- **Normalization Modules**: Multiple normalization strategies (MinMax, Z-Score, Robust)
- **Hugging Face Integration**: Direct loading from Hugging Face Hub repositories
- **Geographic Clustering**: Balanced sampling based on geographic distribution
- **Positional Encoding**: Built-in support for positional embeddings
- **Flexible Patch Modes**: Rectangular and parabolic patch extraction
- **Lazy Loading**: Memory-efficient coordinate generation and chunk caching

## Installation

### Using PDM (Recommended)

```bash
cd /path/to/dataloader
pdm install
```

### Using pip

```bash
pip install -e .
```

### With optional dependencies

```bash
# For Jupyter environment
pdm install -G jupyter_env

# For geospatial features
pdm install -G geospatial

# For development
pdm install -G dev

# All optional dependencies
pdm install -G :all
```

## Quick Start

### Basic Usage

```python
from maya4 import get_sar_dataloader, SARTransform

# Create normalization transform
transforms = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=-3000,
    rc_max=3000,
    gt_min=-12000,
    gt_max=12000,
    complex_valued=True
)

# Create dataloader
loader = get_sar_dataloader(
    data_dir='/path/to/sar_data',
    level_from='rcmc',
    level_to='az',
    batch_size=16,
    num_workers=4,
    patch_size=(1000, 1),
    stride=(300, 1),
    transform=transforms,
    online=True,
    max_products=10
)

# Iterate over batches
for x_batch, y_batch in loader:
    print(f'Input shape: {x_batch.shape}, Target shape: {y_batch.shape}')
    # Your training loop here
```

### Advanced Configuration

```python
from maya4 import SampleFilter, get_sar_dataloader

# Create filter for specific data
filters = SampleFilter(
    years=[2023],
    polarizations=['hh'],
    stripmap_modes=[1, 2, 3],
    parts=['PT1', 'PT3']
)

# Create dataloader with filters
loader = get_sar_dataloader(
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

## Package Structure

```
maya4/
├── __init__.py           # Package initialization and public API
├── dataloader.py         # Main dataloader implementation
├── normalization.py      # Normalization and transformation modules
├── api.py                # Hugging Face Hub API integration
├── utils.py              # Utility functions and helpers
├── location_utils.py     # Geographic location utilities
└── dataset_creation.py   # Dataset creation and Zarr utilities
```

## Main Components

### SARZarrDataset

PyTorch Dataset for loading SAR data patches from Zarr format archives.

**Features:**
- Efficient patch sampling with multiple modes (rectangular, parabolic)
- Chunk-level LRU caching for fast repeated access
- Support for both local and remote (Hugging Face) Zarr stores
- Flexible filtering by part, year, month, polarization, stripmap mode
- Positional encoding support
- Concatenation of patches for transformer models

### SARDataloader

Custom DataLoader with `KPatchSampler` for balanced sampling across products.

**Features:**
- Configurable samples per product
- File and patch shuffling
- Support for different patch orders (row, col, chunk)
- Geographic clustering for balanced sampling

### SARTransform

Modular transformation system with multiple normalization strategies.

**Supported Normalizations:**
- **MinMax**: Normalize to [0, 1] range
- **Z-Score**: Standardization using mean and std
- **Robust**: Median and IQR-based normalization
- **Adaptive**: Compute statistics from data on-the-fly

### API Utilities

Functions for interacting with Hugging Face Hub:
- `list_base_files_in_repo`: List files in a repository
- `fetch_chunk_from_hf_zarr`: Download specific chunks
- `download_metadata_from_product`: Download metadata files

## Configuration Example

```python
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

from maya4 import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config)
```

## Dependencies

### Core Dependencies
- pandas >= 1.5.2
- numpy >= 1.24.0
- torch >= 2.0.0
- zarr >= 2.14.0
- dask[array] >= 2023.5.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- huggingface-hub >= 0.16.0

### Optional Dependencies
- **jupyter_env**: Jupyter notebook support
- **geospatial**: Geographic processing tools
- **dev**: Development and testing tools
- **docs**: Documentation generation

## License

GPLv3

## Authors

Roberto Del Prete - roberto.delprete@esa.int

## Citation

If you use this package in your research, please cite:

```bibtex
@software{maya4_2024,
  author = {Del Prete, Roberto},
  title = {Maya4: SAR Data Processing and Dataloader},
  year = {2024},
  url = {https://github.com/sirbastiano/maya4}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.1.0 (2024-11-26)
- Initial release
- Core dataloader functionality
- Multiple normalization strategies
- Hugging Face Hub integration
- Geographic clustering support
- Positional encoding
- Lazy loading and caching
