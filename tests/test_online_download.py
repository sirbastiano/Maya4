"""
Online mode tests to pre-download data for CI/CD.

This test module is designed to run FIRST in CI/CD pipelines to download
required test data from HuggingFace using online mode. Once data is cached
locally, subsequent offline tests can use it without re-downloading.
"""

import os
from pathlib import Path

import pytest

from maya4.dataloader import get_sar_dataloader
from maya4.utils import SampleFilter


@pytest.mark.online
@pytest.fixture
def data_dir():
    """Get or create the data directory."""
    data_path = Path("/Data/sar_focusing")
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    return str(data_path)


@pytest.mark.online
def test_download_sample_data_online(data_dir):
    """
    Download a small sample of data using online mode.

    This test runs with online=True to download data from HuggingFace.
    Downloaded files will be cached in /Data/sar_focusing for use by
    subsequent offline tests.
    """
    # Configure filters to limit download size
    filters = SampleFilter(
        parts=["PT1"],  # Only one part to minimize download
        years=[2025],  # Recent data
        polarizations=["hh"],  # Single polarization
    )

    # Create dataloader with online mode
    dataloader = get_sar_dataloader(
        data_dir=data_dir,
        filters=filters,
        batch_size=1,
        num_workers=0,  # Sequential for CI stability
        patch_size=(128, 128),
        buffer=(50, 50),
        stride=(100, 100),
        online=True,  # Enable online downloads
        max_products=2,  # Limit to 2 files
        samples_per_prod=5,  # Just a few samples
        use_balanced_sampling=False,
        transform=None,
        positional_encoding=False,
        complex_valued=False,
        verbose=True,
    )

    # Verify we can get files
    files = dataloader.dataset.get_files()
    print(f"Found {len(files)} files available online")

    # Try to load a few batches to trigger download
    batch_count = 0
    for x_batch, y_batch in dataloader:
        batch_count += 1
        assert x_batch is not None, "x_batch should not be None"
        assert y_batch is not None, "y_batch should not be None"
        print(f"Downloaded batch {batch_count}: x={x_batch.shape}, y={y_batch.shape}")

        if batch_count >= 3:  # Just download a few batches
            break

    # Verify files were downloaded
    downloaded_files = list(Path(data_dir).rglob("*.zarr"))
    print(f"Downloaded {len(downloaded_files)} zarr files to {data_dir}")

    # Assert we downloaded something
    assert len(downloaded_files) > 0, "Should have downloaded at least one zarr file"
    assert batch_count > 0, "Should have loaded at least one batch"


@pytest.mark.online
def test_verify_downloaded_data_offline(data_dir):
    """
    Verify downloaded data can be accessed in offline mode.

    This test runs AFTER test_download_sample_data_online to verify
    that downloaded files are accessible without online mode.
    """
    # Check if data was downloaded
    downloaded_files = list(Path(data_dir).rglob("*.zarr"))

    if len(downloaded_files) == 0:
        pytest.skip("No data downloaded in online test, skipping offline verification")

    print(f"Verifying {len(downloaded_files)} downloaded files")

    # Try to access with online=False
    filters = SampleFilter(parts=["PT1"], years=[2025], polarizations=["hh"])

    dataloader = get_sar_dataloader(
        data_dir=data_dir,
        filters=filters,
        batch_size=1,
        num_workers=0,
        patch_size=(128, 128),
        online=False,  # Offline mode
        max_products=2,
        samples_per_prod=3,
        use_balanced_sampling=False,
        transform=None,
        positional_encoding=False,
        verbose=True,
    )

    # Verify we can access files
    files = dataloader.dataset.get_files()
    assert len(files) > 0, "Should find downloaded files in offline mode"
    print(f"Successfully accessed {len(files)} files in offline mode")

    # Load one batch to verify data integrity
    for x_batch, y_batch in dataloader:
        assert x_batch is not None
        assert y_batch is not None
        print(f"Offline batch loaded: x={x_batch.shape}, y={y_batch.shape}")
        break  # Just verify one batch works


if __name__ == "__main__":
    # For local testing
    import sys

    data_dir = "/Data/sar_focusing"

    print("Testing online download...")
    test_download_sample_data_online(data_dir)

    print("\nTesting offline access...")
    test_verify_downloaded_data_offline(data_dir)

    print("\nAll online/offline tests passed!")
