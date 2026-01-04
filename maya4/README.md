# SAR Dataloader Documentation

## Overview

The Maya4 SAR (Synthetic Aperture Radar) dataloader provides a high-performance, flexible PyTorch-compatible data loading pipeline for SAR imagery stored in Zarr format. It supports both local and remote (HuggingFace) data sources, with advanced features like lazy loading, chunk-level caching, positional encoding, and various normalization strategies.

## Architecture

### Core Components

```
maya4/
├── dataloader.py           # Main dataset and dataloader classes
├── caching.py             # LRU chunk cache implementation
├── coords_generation.py   # Lazy coordinate generation
├── positional_encoding.py # Positional encoding modules
├── normalization.py       # Data normalization transforms
└── utils.py              # Utility functions and filters
```

### Component Relationships

```
SARDataloader
    └── SARZarrDataset
        ├── ChunkCache (caching.py)
        │   └── LRU cache for zarr chunks
        ├── LazyCoordinateGenerator (coords_generation.py)
        │   └── Memory-efficient coordinate iteration
        ├── PositionalEncoding (positional_encoding.py)
        │   ├── PositionalEncoding2D (2D spatial encoding)
        │   └── PositionalEncodingRow (row-only encoding)
        └── SARTransform (normalization.py)
            ├── MinMaxNormalize
            ├── ZScoreNormalize
            ├── RobustNormalize
            └── AdaptiveNormalizationModule
```

## Quick Start

### Basic Usage

```python
from maya4.dataloader import get_sar_dataloader

# Create dataloader with default settings
dataloader = get_sar_dataloader(
    data_dir="/path/to/sar_data",
    batch_size=16,
    num_workers=4,
    online=False  # Use local files
)

# Iterate over batches
for x_batch, y_batch in dataloader:
    # x_batch: input patches (e.g., RCMC level)
    # y_batch: target patches (e.g., AZ level)
    print(f"Input shape: {x_batch.shape}")
    print(f"Target shape: {y_batch.shape}")
```

### Advanced Configuration

```python
from maya4.dataloader import get_sar_dataloader
from maya4.normalization import SARTransform
from maya4.utils import SampleFilter, RC_MIN, RC_MAX, GT_MIN, GT_MAX

# Configure data filtering
filters = SampleFilter(
    years=[2023, 2024],
    polarizations=["hh", "vv"],
    stripmap_modes=[1, 2, 3],
    parts=["PT1", "PT2"]
)

# Configure normalization
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=RC_MIN,
    rc_max=RC_MAX,
    gt_min=GT_MIN,
    gt_max=GT_MAX,
    complex_valued=True
)

# Create dataloader with custom settings
dataloader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    filters=filters,
    transform=transform,
    batch_size=32,
    num_workers=8,
    patch_size=(256, 512),      # Height x Width
    buffer=(100, 100),          # Edge buffer
    stride=(128, 256),          # Patch stride
    level_from="rcmc",          # Input SAR level
    level_to="az",              # Target SAR level
    positional_encoding="2d",   # Add 2D positional encoding
    complex_valued=False,       # Return real/imag stacked
    cache_size=10000,           # Chunk cache size
    online=False,               # Local mode
    max_products=50,            # Limit number of files
    samples_per_prod=1000,      # Patches per file
    shuffle_files=True,         # Shuffle file order
    patch_order="row",          # Row-major patch order
    use_balanced_sampling=True, # Balance dataset splits
    split="train"               # Dataset split
)
```

## Core Classes

### 1. SARZarrDataset

Main PyTorch Dataset class for SAR data.

**Key Features:**
- Lazy loading of zarr stores (opened on-demand)
- Memory-efficient coordinate generation
- Chunk-level LRU caching
- Support for local and remote (HuggingFace) data
- Flexible patch extraction with configurable size, stride, and buffer
- Optional positional encoding and data normalization

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | required | Directory containing zarr files |
| `filters` | SampleFilter | None | Filter for data selection |
| `author` | str | "Maya4" | HuggingFace author for online mode |
| `online` | bool | False | Enable remote data access |
| `transform` | SARTransform | None | Normalization transform |
| `patch_size` | Tuple[int,int] | (256,256) | Patch dimensions (H,W) |
| `complex_valued` | bool | False | Return complex or real/imag stacked |
| `level_from` | str | "rcmc" | Input SAR processing level |
| `level_to` | str | "az" | Target SAR processing level |
| `buffer` | Tuple[int,int] | (100,100) | Edge buffer to avoid |
| `stride` | Tuple[int,int] | (50,50) | Patch sampling stride |
| `cache_size` | int | 1000 | Number of chunks to cache |
| `positional_encoding` | str | "2d" | Type: "2d", "row", or "none" |
| `max_products` | int | 10 | Maximum files to use |
| `samples_per_prod` | int | 1000 | Patches per file |
| `verbose` | bool | True | Print debug information |

**Special Patch Size Values:**
- Use `-1` for height or width to use full image dimension
- Example: `patch_size=(1, -1)` extracts full rows

**Methods:**

```python
# Get list of files
files = dataset.get_files()

# Get coordinates for a specific file
coords = dataset.get_samples_by_file(file_path)

# Visualize a patch
dataset.visualize_item(
    idx=(file_path, y, x),
    show=True,
    vminmax=(0, 1000)
)

# Get patch size (resolves -1 values)
height, width = dataset.get_patch_size(file_path)

# Get full image dimensions
height, width = dataset.get_whole_sample_shape(file_path)
```

### 2. KPatchSampler

PyTorch Sampler for efficient patch sampling.

**Features:**
- Round-robin sampling from multiple files
- Optional file and patch shuffling
- Configurable patch order (row, column, chunk)
- Per-file sample limiting

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | SARZarrDataset | required | Dataset to sample from |
| `samples_per_prod` | int | 0 | Patches per file (0 = all) |
| `shuffle_files` | bool | True | Shuffle file order |
| `patch_order` | str | "row" | "row", "col", or "chunk" |
| `seed` | int | 42 | Random seed |
| `verbose` | bool | True | Print debug info |

**Methods:**

```python
# Filter to specific files
sampler.filter_by_zfiles([file1, file2, file3])

# Get coordinates for a file
coords = sampler.get_coords_from_store(file_path)
```

### 3. SARDataloader

Extended PyTorch DataLoader with SAR-specific features.

**Additional Methods:**

```python
# Get coordinates for a file with optional window
coords = dataloader.get_coords_from_zfile(
    zfile=file_path,
    window=((y_start, y_end), (x_start, x_end))
)

# Filter to specific files
dataloader.filter_by_zfiles([file1, file2, file3])
```

## Component Details

### Caching System (ChunkCache)

The caching system provides LRU (Least Recently Used) caching at the chunk level for efficient repeated access.

**Architecture:**
```
ChunkCache
├── _load_chunk_uncached()      # Load from zarr or HuggingFace
├── _load_chunk()                # LRU-cached chunk loading
├── get_sample()                 # Main entry point
└── Optimized paths:
    ├── _get_horizontal_strip_optimized()  # For (1, W) patches
    ├── _get_vertical_strip_optimized()    # For (H, 1) patches
    ├── _get_small_patch_optimized()       # For patches < chunk
    └── _get_sample_from_cached_chunks()   # General case
```

**Features:**
- Automatic HuggingFace download for online mode
- Optimized loading for special patch sizes (full rows/columns)
- Memory-efficient chunk reuse
- Configurable cache size

**Usage:**
```python
# Cache is automatically managed by the dataset
dataset = SARZarrDataset(
    data_dir="...",
    cache_size=10000,  # Number of chunks to cache
    online=True        # Enable online chunk downloading
)
```

### Coordinate Generation (LazyCoordinateGenerator)

Memory-efficient coordinate generation without materializing arrays.

**Classes:**

1. **LazyCoordinateRange**: Represents a range with start, stop, step
2. **LazyCoordinateGenerator**: Generates (y, x) coordinates on-the-fly

**Patch Orders:**
- `"row"`: Row-major order (iterate x within each y)
- `"col"`: Column-major order (iterate y within each x)
- `"chunk"`: Block-based iteration (useful for spatial locality)

**Example:**
```python
from maya4.coords_generation import LazyCoordinateRange, LazyCoordinateGenerator

# Create coordinate ranges
y_range = LazyCoordinateRange(0, 1000, 10)  # 0, 10, 20, ..., 990
x_range = LazyCoordinateRange(0, 2000, 20)  # 0, 20, 40, ..., 1980

# Generate coordinates
gen = LazyCoordinateGenerator(
    y_range=y_range,
    x_range=x_range,
    patch_order="row"
)

# Iterate (no memory overhead)
for y, x in gen:
    print(f"Patch at ({y}, {x})")
```

### Positional Encoding

Adds spatial position information to patches for transformer-based models.

**Available Encodings:**

1. **PositionalEncoding2D** (`"2d"`):
   - Adds 2 channels (row and column encodings)
   - Sine/cosine encoding for both spatial dimensions
   - Output: `(H, W, C+2)` where C is input channels

2. **PositionalEncodingRow** (`"row"`):
   - Adds 1 channel (row encoding only)
   - Useful for 1D transformers or row-wise processing
   - Output: `(H, W, C+1)`

3. **None** (`False` or `"none"`):
   - No positional encoding
   - Output: `(H, W, C)` unchanged

**Factory Function:**
```python
from maya4.positional_encoding import create_positional_encoding_module

# Create 2D encoding
pe_2d = create_positional_encoding_module(
    name="2d",
    complex_valued=False,
    concat=True  # Concatenate to input (vs replace)
)

# Create row encoding
pe_row = create_positional_encoding_module(
    name="row",
    complex_valued=True,
    concat=True
)

# Apply encoding
encoded = pe_2d.forward(
    patch,                        # Input array
    position=(y, x),              # Current position
    max_length=(max_h, max_w)    # Maximum dimensions
)
```

**Configuration in Dataloader:**
```python
# With positional encoding
dataloader = get_sar_dataloader(
    data_dir="...",
    positional_encoding="2d",  # or "row" or False
    complex_valued=False       # Must match encoding setting
)

# Without positional encoding
dataloader = get_sar_dataloader(
    data_dir="...",
    positional_encoding=False  # or None
)
```

### Normalization (SARTransform)

Flexible data normalization with forward and inverse transforms.

**Available Normalizations:**

1. **MinMaxNormalize**: Scale to [0, 1] or custom range
2. **ZScoreNormalize**: Standardize to zero mean, unit variance
3. **RobustNormalize**: Robust scaling using percentiles
4. **AdaptiveNormalizationModule**: Context-dependent normalization

**Factory Methods:**

```python
from maya4.normalization import SARTransform
from maya4.utils import RC_MIN, RC_MAX, GT_MIN, GT_MAX

# MinMax normalization
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=RC_MIN,      # Min for RCMC level
    rc_max=RC_MAX,      # Max for RCMC level
    gt_min=GT_MIN,      # Min for ground truth level
    gt_max=GT_MAX,      # Max for ground truth level
    complex_valued=True
)

# Z-score normalization
transform = SARTransform.create_zscore_normalized_transform(
    normalize=True,
    rc_mean=0.0,
    rc_std=1.0,
    gt_mean=0.0,
    gt_std=1.0,
    complex_valued=True
)

# Robust normalization
transform = SARTransform.create_robust_normalized_transform(
    normalize=True,
    percentile_range=(1, 99),  # Use 1st-99th percentile
    complex_valued=True
)
```

**Using Transforms:**

```python
# Apply transform (automatically called by dataset)
normalized = transform(patch, level="rcmc")

# Inverse transform (restore original values)
original = transform.inverse(normalized, level="rcmc")

# Round-trip test
patch_orig = np.random.randn(64, 128) + 1j * np.random.randn(64, 128)
normalized = transform(patch_orig, "rcmc")
restored = transform.inverse(normalized, "rcmc")
error = np.abs(restored - patch_orig).max()  # Should be ~1e-12
```

**Level-Specific Normalization:**
The transform automatically selects the correct normalization based on the SAR processing level:
- `"rcmc"`, `"rc"`, etc. → Use `rc_min/rc_max` or `rc_mean/rc_std`
- `"az"`, `"ground_truth"`, etc. → Use `gt_min/gt_max` or `gt_mean/gt_std`

## Data Organization

### Directory Structure

**Local Mode:**
```
/Data/sar_focusing/
├── PT1/
│   ├── file1.zarr/
│   │   ├── rcmc/
│   │   ├── az/
│   │   └── ...
│   ├── file2.zarr/
│   └── ...
├── PT2/
├── PT3/
└── PT4/
```

**Remote Mode (HuggingFace):**
```
HuggingFace Hub:
├── author/PT1  (repository)
│   ├── file1.zarr
│   ├── file2.zarr
│   └── ...
├── author/PT2
├── author/PT3
└── author/PT4
```

### Zarr File Structure

Each `.zarr` file contains multiple processing levels:
```
product.zarr/
├── rcmc/          # Range Cell Migration Corrected
├── az/            # Azimuth compressed
├── rc/            # Range Compressed
├── raw/           # Raw data
└── metadata.json  # Product metadata
```

## Filtering and Selection

### SampleFilter

Filter data by various criteria:

```python
from maya4.utils import SampleFilter

# Create filter
filters = SampleFilter(
    years=[2023, 2024, 2025],           # Acquisition years
    months=[1, 2, 3, 6, 7, 8],          # Acquisition months
    polarizations=["hh", "vv"],         # Polarization modes
    stripmap_modes=[1, 2, 3],           # Stripmap modes
    parts=["PT1", "PT2", "PT3", "PT4"]  # Data partitions
)

# Use with dataloader
dataloader = get_sar_dataloader(
    data_dir="...",
    filters=filters
)
```

### SampleFilterRegex

Filter data using regex patterns to match specific product names:

```python
from maya4.utils import SampleFilterRegex

# Create regex filter - matches products containing specific patterns
filter = SampleFilterRegex(
    regex_list=[
        "s1a-s1-raw-s-hh-.*",  # Match all S1A stripmap HH products
        ".*-20241008.*",        # Match products from specific date
        ".*-056012-.*"          # Match specific orbit number
    ]
)

# Or match exact product names
filter = SampleFilterRegex(
    regex_list=["s1a-s1-raw-s-hh-20241008t151239-20241008t151255-056012-06d9c9.zarr"]
)

# Use with dataloader
dataloader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    filters=filter,
    online=False
)
```

**Key differences from SampleFilter:**
- `SampleFilterRegex`: Pattern-based matching using regular expressions for specific products selection
- `SampleFilter`: Structured filtering by metadata fields (years, months, polarizations, etc.)

Use `SampleFilterRegex` when you need to:
- Match specific product filenames
- Filter by complex patterns (e.g., date ranges, orbit numbers)
- Select products that don't fit structured metadata criteria

### Balanced Sampling

Ensure balanced representation across splits:

```python
dataloader = get_sar_dataloader(
    data_dir="...",
    use_balanced_sampling=True,  # Enable balanced sampling
    split="train",                # "train", "val", or "test"
    max_products=100              # Total products to select
)
```

## Performance Optimization

### Memory Efficiency

1. **Lazy Loading**: Zarr stores opened on-demand
2. **Lazy Coordinates**: No array materialization
3. **Chunk Caching**: LRU cache reuses chunks
4. **Streaming**: Iterate without loading all data

### Speed Optimization

1. **Increase Cache Size**:
   ```python
   dataloader = get_sar_dataloader(
       data_dir="...",
       cache_size=50000  # Larger cache for more reuse
   )
   ```

2. **Increase Workers**:
   ```python
   dataloader = get_sar_dataloader(
       data_dir="...",
       num_workers=16  # More parallel loading
   )
   ```

3. **Optimize Patch Size**:
   ```python
   # Align with chunk size for efficiency
   dataloader = get_sar_dataloader(
       data_dir="...",
       patch_size=(256, 256),  # Match zarr chunks
       stride=(256, 256)        # Non-overlapping
   )
   ```

4. **Use Optimized Patch Modes**:
   ```python
   # Full rows are optimized
   dataloader = get_sar_dataloader(
       data_dir="...",
       patch_size=(1, -1),  # Full row extraction
       stride=(1, 1)
   )
   ```

## Common Use Cases

### 1. Training a Model

```python
from maya4.dataloader import get_sar_dataloader
from maya4.normalization import SARTransform
from maya4.utils import RC_MIN, RC_MAX, GT_MIN, GT_MAX

# Setup
transform = SARTransform.create_minmax_normalized_transform(
    normalize=True,
    rc_min=RC_MIN, rc_max=RC_MAX,
    gt_min=GT_MIN, gt_max=GT_MAX,
    complex_valued=False
)

train_loader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    transform=transform,
    batch_size=32,
    num_workers=8,
    patch_size=(256, 256),
    complex_valued=False,
    positional_encoding="2d",
    split="train",
    use_balanced_sampling=True
)

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        # x_batch: (B, H, W, 4) - 2 for real/imag + 2 for PE
        # y_batch: (B, H, W, 4)
        
        # Move to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### 2. Data Validation

```python
from maya4.dataloader import get_sar_dataloader

# Load without transforms
loader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    transform=None,
    batch_size=1,
    num_workers=0,
    patch_size=(128, 128),
    positional_encoding=False,
    max_products=1,
    samples_per_prod=10
)

# Check data integrity
for i, (x, y) in enumerate(loader):
    print(f"Batch {i}: x={x.shape}, y={y.shape}")
    print(f"  x range: [{x.min():.2e}, {x.max():.2e}]")
    print(f"  y range: [{y.min():.2e}, {y.max():.2e}]")
    
    # Visualize
    if i == 0:
        loader.dataset.visualize_item(
            idx=(file, y, x),
            show=True
        )
```

### 3. Full Row/Column Processing

```python
# Process full rows
row_loader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    patch_size=(1, -1),  # Full width
    buffer=(0, 0),
    stride=(1, 1),       # Every row
    batch_size=8,
    positional_encoding=False
)

# Process full columns
col_loader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    patch_size=(-1, 1),  # Full height
    buffer=(0, 0),
    stride=(1, 1),       # Every column
    batch_size=8,
    positional_encoding=False
)
```

### 4. Online Mode with HuggingFace

```python
# Download and process remote data
online_loader = get_sar_dataloader(
    data_dir="/local/cache/dir",
    author="Maya4",          # HuggingFace author
    online=True,             # Enable remote access
    batch_size=16,
    cache_size=20000,        # Larger cache for remote
    max_products=20
)

# Data automatically downloaded on-demand
for x, y in online_loader:
    process_batch(x, y)
```

## Testing and Validation

### Data Integrity Tests

The test suite includes comprehensive data integrity validation:

```python
# Run all tests
pytest tests/test_dataloader_clean.py tests/test_data_integrity_detailed.py -v

# Test specific functionality
pytest tests/test_dataloader_clean.py::TestDataloaderAccuracy -v
pytest tests/test_data_integrity_detailed.py -v
```

**Test Coverage:**
- Raw data loading (perfect bit-for-bit match with zarr)
- Normalization forward/inverse (picometer-level precision)
- Positional encoding (original data preservation)
- Horizontal/vertical/rectangular patch extraction
- Caching correctness
- Lazy coordinate generation

## Troubleshooting

### Common Issues

1. **No files found**:
   ```python
   # Check filter settings
   filters = SampleFilter(years=[2023, 2024])  # Too restrictive?
   
   # Check data directory
   print(dataloader.dataset.get_files())  # Should return files
   ```

2. **Slow initialization**:
   ```python
   # Reduce max_products or use balanced sampling
   dataloader = get_sar_dataloader(
       data_dir="...",
       max_products=10,  # Start small
       use_balanced_sampling=False  # Skip if slow
   )
   ```

3. **Memory errors**:
   ```python
   # Reduce cache size and batch size
   dataloader = get_sar_dataloader(
       data_dir="...",
       cache_size=1000,  # Smaller cache
       batch_size=8      # Smaller batches
   )
   ```

4. **None batches**:
   - Ensure `positional_encoding` is correctly set
   - Use `False` or valid string ("2d", "row")
   - Don't use `True` (not supported)

## API Reference

See inline documentation in:
- `maya4/dataloader.py` - Main classes
- `maya4/caching.py` - Caching system
- `maya4/coords_generation.py` - Coordinate generation
- `maya4/positional_encoding.py` - Positional encoding
- `maya4/normalization.py` - Normalization transforms

## Examples

Complete examples available in:
- `notebooks/dataloader.ipynb` - Interactive usage
- `tests/test_data_integrity_detailed.py` - Validation examples
- `dataloader.py` (main section) - Basic usage

## Version Information

- PyTorch: 2.0+
- Zarr: 3.0+ (breaking changes from v2)
- Pandas: 1.5+
- NumPy: 1.24+

## License

See LICENSE file in repository root.
