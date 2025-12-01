"""
Pytest configuration and shared fixtures for maya4 tests.
"""
import pytest
import numpy as np
import zarr
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_zarr_store(temp_dir):
    """Create a sample Zarr store for testing."""
    zarr_path = temp_dir / "test_sample.zarr"
    
    # Create sample data
    height, width = 1000, 500
    
    # Create Zarr group
    store = zarr.open_group(str(zarr_path), mode='w')
    
    # Add sample arrays for different processing levels
    rcmc_data = np.random.randn(height, width) + 1j * np.random.randn(height, width)
    store.create_array(
        'rcmc',
        shape=(height, width),
        chunks=(128, 128),
        dtype='complex64'
    )
    store['rcmc'][:] = rcmc_data
    
    az_data = np.random.randn(height, width) + 1j * np.random.randn(height, width)
    store.create_array(
        'az',
        shape=(height, width),
        chunks=(128, 128),
        dtype='complex64'
    )
    store['az'][:] = az_data
    
    rc_data = np.random.randn(height, width) + 1j * np.random.randn(height, width)
    store.create_array(
        'rc',
        shape=(height, width),
        chunks=(128, 128),
        dtype='complex64'
    )
    store['rc'][:] = rc_data
    
    # Add metadata
    store.attrs['parent_product'] = 'S1A_IW_SLC__1SSH_20250101T000000_stripmap_mode_1'
    store.attrs['creation_date'] = '2025-01-01T00:00:00'
    
    return zarr_path


@pytest.fixture
def sample_filter():
    """Create a sample filter for testing."""
    from maya4.utils import SampleFilter
    return SampleFilter(
        parts=['part1'],
        years=[2024],
        polarizations=['HH']
    )


@pytest.fixture
def sample_transform():
    """Create a sample transform for testing."""
    from maya4.normalization import SARTransform
    from maya4.utils import RC_MIN, RC_MAX, GT_MIN, GT_MAX
    
    return SARTransform.create_minmax_normalized_transform(
        normalize=True,
        rc_min=RC_MIN,
        rc_max=RC_MAX,
        gt_min=GT_MIN,
        gt_max=GT_MAX,
        complex_valued=True
    )
