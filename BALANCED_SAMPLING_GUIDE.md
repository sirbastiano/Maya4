# Understanding `get_balanced_sample_files()`

## Purpose

This function selects a balanced subset of SAR product files ensuring **equal representation** across geographic locations and scene types (land, sea, coast).

## Why Use It?

When training ML models on SAR data, you want:
1. **Balanced scene types** - Not 90% sea and 10% land
2. **Geographic diversity** - Samples from different locations
3. **No overlap** - Avoid redundant geographic coverage
4. **Proper splits** - Respect train/validation/test boundaries

## Function Signature

```python
get_balanced_sample_files(
    max_samples: int,                    # Total samples to select
    data_dir: str,                       # Path to SAR data
    sample_filter: Optional[SampleFilter] = None,  # Optional filter
    config_path: Optional[str] = None,   # Path to split configs
    split_type: str = "train",           # 'train', 'validation', or 'test'
    ensure_representation: bool = True,  # Enable balanced sampling
    min_samples_per_cluster: int = 1,    # Min per geographic cluster
    verbose: bool = False,               # Show progress
    n_clusters: int = 20,                # Number of geographic clusters
    repo_author: str = 'Maya4',          # HuggingFace author
    repos: List[str] = ['PT1', ...]      # Repositories to use
) -> List[str]
```

## How It Works

### Step 1: Load Dataset Splits

The function looks for pre-computed CSV files like:
- `dataset_splits/train_products.csv`
- `dataset_splits/validation_products.csv`
- `dataset_splits/test_products.csv`

These files contain:
```csv
filename;part;geo_cluster;scene_type;coordinates;year;month
s1a-s1-raw-s-hh-....zarr;PT1;5;land;[[[lon,lat],...]]];2024;10
```

If they don't exist, the function creates them by:
1. Calling `get_products_spatial_mapping()` to get locations
2. Running `create_balanced_splits()` to cluster products geographically

### Step 2: Apply Filters (Optional)

If you provide a `SampleFilter`, it filters by:
```python
sample_filter = SampleFilter(
    years=[2024, 2025],          # Only these years
    months=[1, 2, 3],             # Only Jan, Feb, Mar
    polarizations=['hh'],         # Only HH polarization
    stripmap_modes=[1, 2],        # Only modes 1 and 2
    parts=['PT1', 'PT2']          # Only these partitions
)
```

### Step 3: Balanced Sampling Strategy

The function aims for **EQUAL distribution** across scene types:

```
Target: 30 samples
├─ Land:  10 samples (33.3%)
├─ Sea:   10 samples (33.3%)
└─ Coast: 10 samples (33.3%)
```

**Algorithm:**

1. **Remove duplicates** - Drop products at same geographic location
2. **Initial allocation** - Try to get `max_samples/3` from each scene type
3. **Handle deficit** - If one type lacks samples:
   - Use **round-robin** filling from other types
   - Cycle through [land, sea, coast] fairly
   - Avoid geographic overlap using polygon intersection
4. **Relaxed mode** - If strict overlap fails, relax constraints

### Step 4: Overlap Avoidance

Each product has a geographic polygon (bounding box). The function:
1. Parses coordinates from CSV
2. Creates Shapely Polygon objects
3. Checks intersection between polygons
4. Only selects non-overlapping samples

```python
# Check if two products overlap
if polygon_a.intersects(polygon_b):
    intersection = polygon_a.intersection(polygon_b)
    if intersection.area > 0.0001:
        # Products overlap - skip
```

## Example Usage

### Example 1: Basic Balanced Sampling

```python
from maya4.utils import get_balanced_sample_files

files = get_balanced_sample_files(
    max_samples=30,
    data_dir="/Data/sar_focusing",
    split_type="train",
    ensure_representation=True,
    verbose=True
)

# Output: 30 files with ~10 land, ~10 sea, ~10 coast
```

### Example 2: With Polarization Filter

```python
from maya4.utils import SampleFilter, get_balanced_sample_files

# Only HH polarization
filter_hh = SampleFilter(polarizations=['hh'])

files = get_balanced_sample_files(
    max_samples=20,
    data_dir="/Data/sar_focusing",
    sample_filter=filter_hh,
    split_type="train",
    verbose=True
)

# Output: 20 HH files with balanced scene types
```

### Example 3: Specific Year and Region

```python
# Only 2024 data from PT1
filter_2024 = SampleFilter(
    years=[2024],
    parts=['PT1']
)

files = get_balanced_sample_files(
    max_samples=15,
    data_dir="/Data/sar_focusing",
    sample_filter=filter_2024,
    split_type="validation",
    repos=['PT1'],
    verbose=True
)
```

### Example 4: Integration with Dataloader

```python
from maya4 import get_sar_dataloader, SampleFilter
from maya4.utils import get_balanced_sample_files

# Get balanced files
files = get_balanced_sample_files(
    max_samples=100,
    data_dir="/Data/sar_focusing",
    split_type="train",
    verbose=True
)

# Create filter from selected files
# Extract just filenames
filenames = [Path(f).name for f in files]

# Create regex filter
from maya4.utils import SampleFilterRegex
filter_balanced = SampleFilterRegex(regex_list=filenames)

# Use with dataloader
loader = get_sar_dataloader(
    data_dir="/Data/sar_focusing",
    level_from="rcmc",
    level_to="az",
    filters=filter_balanced,
    batch_size=16,
    online=False
)
```

## Verbose Output Explanation

When `verbose=True`, you see:

```
Getting balanced sample of 30 files from train set...
  Loading existing split file: .../train_products.csv
Loaded 500 products from train split

Balanced sampling with EQUAL scene type representation:
  Total available products (after filters): 500
  Dropped 50 duplicate positions
  Remaining unique positions: 450

  Scene type distribution after deduplication:
    land: 200 products (44.4%)
    sea: 180 products (40.0%)
    coast: 70 products (15.6%)

  Target: 30 total samples
  Equal allocation: 10 samples per scene type

    land: 10/10 samples selected (200 available)
    sea: 10/10 samples selected (180 available)
    coast: 10/10 samples selected (70 available)

  Final distribution:
    land: 10 samples (33.3%, target: 33.3%, Δ+0.0%)
    sea: 10 samples (33.3%, target: 33.3%, Δ+0.0%)
    coast: 10 samples (33.3%, target: 33.3%, Δ+0.0%)

  Total selected: 30 samples
```

## Key Insights

### Round-Robin Filling Strategy

**Problem:** Old approach sorted by available samples (descending), so the most abundant class (usually sea) filled all deficits first.

**Solution:** New approach uses **round-robin**:
```python
while deficit > 0:
    for scene_type in ['land', 'sea', 'coast']:
        # Try to add ONE sample from this type
        # Then move to next type
        # This ensures fair distribution
```

### Overlap Avoidance Levels

1. **Strict** - No polygon intersection (area > 0.0001)
2. **Relaxed** - Allow small overlaps if needed to reach target

### When It Falls Back

The function falls back to random sampling when:
- No scene type information in CSV
- No coordinate information in CSV
- All samples filtered out
- Cannot create balanced distribution

## Common Issues

### Issue 1: "Split file not found"

**Cause:** First time running, no pre-computed splits exist.

**Solution:** Function automatically creates them by:
1. Scanning HuggingFace repos for products
2. Clustering by geography
3. Creating train/val/test splits (60%/20%/20%)

### Issue 2: "Not enough samples"

**Cause:** After filtering, not enough samples for balanced distribution.

**Solution:**
- Reduce `max_samples`
- Relax filters (fewer constraints)
- Use more repos (`repos=['PT1', 'PT2', 'PT3', 'PT4']`)

### Issue 3: "Unbalanced distribution"

**Cause:** One scene type very rare (e.g., only 5 coast samples).

**Solution:** Function will:
1. Take all available rare samples
2. Fill remainder with round-robin from other types
3. Print warning about imbalance

## Testing

Run the test script:

```bash
python test_balanced_simple.py
```

This will show you:
- How balanced sampling works
- Effect of filters
- Distribution statistics
- Sample output

## Performance

- **Fast** if splits already exist (~1-2 seconds)
- **Slow** first time (creates splits, ~5-10 minutes for all repos)
- **Cached** - Splits saved to disk for reuse

## Related Functions

- `get_products_spatial_mapping()` - Gets product locations from HuggingFace
- `create_balanced_splits()` - Creates geographic clusters and splits
- `SampleFilter` - Filters by metadata (year, month, polarization, etc.)
- `SampleFilterRegex` - Filters by filename patterns
- `_get_balanced_representation_samples()` - Internal balanced sampling logic
- `_select_non_overlapping_samples()` - Internal overlap avoidance logic
