"""
Tests for SARZarrDataset class in dataloader_clean module.

NOTE: This test file is EXCLUDED from CI/CD (see .github/workflows/tests.yml).
It was written for the old dataloader_clean.py module which has been consolidated
into dataloader.py. The functionality is now tested more comprehensively in:
- test_dataloader_clean.py (24 tests covering all SARZarrDataset functionality)
- test_data_integrity_detailed.py (4 tests with detailed validation)

This file is kept for reference but should not be run in CI as it has mock issues
and is redundant with the newer, more comprehensive test suite.

Tests cover:
- Local file discovery and initialization
- Remote (online) file discovery
- File filtering with SampleFilter
- DataFrame structure and metadata
- Lazy coordinate generation
- Patch extraction and caching
"""

import pytest
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from maya4.dataloader import SARZarrDataset, get_sar_dataloader
from maya4.utils import SampleFilter
from maya4.coords_generation import LazyCoordinateGenerator


class TestSARZarrDatasetLocal:
    """Tests for SARZarrDataset with online=False (local files)."""
    
    def test_initialization_empty_directory(self, temp_dir):
        """Test initialization with empty directory."""
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False
        )
        
        assert dataset._files is not None
        assert len(dataset.get_files()) == 0
        assert len(dataset) == 0
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_local_file_discovery(self, mock_parse, temp_dir):
        """Test that local zarr files are discovered correctly."""
        # Create mock zarr structure
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        zarr_file1 = part_dir / "S1A_IW_SLC__1SDV_20200101T000000_20200101T000001_030001_000001_0001.zarr"
        zarr_file2 = part_dir / "S1A_IW_SLC__1SDV_20200102T000000_20200102T000001_030002_000002_0001.zarr"
        
        # Create actual zarr stores with data
        for zf in [zarr_file1, zarr_file2]:
            zf.mkdir()
            (zf / "rcmc").mkdir()
            (zf / "az").mkdir()
            zarr.open_array(str(zf / "rcmc"), mode='w', shape=(1000, 1000), chunks=(100, 100), dtype=np.complex64)
            zarr.open_array(str(zf / "az"), mode='w', shape=(1000, 1000), chunks=(100, 100), dtype=np.complex64)
        
        # Mock parse function to return metadata
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101',
                'stripmap_mode': 0
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False,
            max_products=10
        )
        
        # Verify files were discovered
        assert dataset._files is not None
        assert len(dataset.get_files()) == 2
        assert 'full_name' in dataset._files.columns
        assert 'part' in dataset._files.columns
        assert 'store' in dataset._files.columns
        assert 'samples' in dataset._files.columns
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_file_filtering(self, mock_parse, temp_dir):
        """Test that SampleFilter correctly filters files."""
        # Create mock zarr files
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        
        zarr_files = []
        for i in range(5):
            zf = part_dir / f"S1A_IW_SLC__1SDV_2020010{i}T000000_2020010{i}T000001_03000{i}_00000{i}_0001.zarr"
            zf.mkdir()
            (zf / "rcmc").mkdir()
            (zf / "az").mkdir()
            zarr.open_array(str(zf / "rcmc"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
            zarr.open_array(str(zf / "az"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
            zarr_files.append(zf)
        
        def parse_side_effect(filepath):
            idx = int(Path(filepath).name.split('_')[6][-1])
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV' if idx % 2 == 0 else 'VH',
                'acquisition_date': f'2020010{idx}',
                'stripmap_mode': idx
            }
        mock_parse.side_effect = parse_side_effect
        
        # Create filter for VV polarization only
        sample_filter = SampleFilter(polarizations=['VV'])
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            filters=sample_filter,
            online=False,
            verbose=False,
            max_products=10
        )
        
        # Should have filtered to only VV files (indices 0, 2, 4)
        files = dataset.get_files()
        assert len(files) == 3
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_max_products_limit(self, mock_parse, temp_dir):
        """Test that max_products limits the number of files."""
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        
        # Create 10 zarr files
        for i in range(10):
            zf = part_dir / f"file_{i}.zarr"
            zf.mkdir()
            (zf / "rcmc").mkdir()
            (zf / "az").mkdir()
            zarr.open_array(str(zf / "rcmc"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
            zarr.open_array(str(zf / "az"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False,
            max_products=3,
            use_balanced_sampling=False
        )
        
        assert len(dataset.get_files()) == 3


class TestSARZarrDatasetOnline:
    """Tests for SARZarrDataset with online=True (remote files)."""
    
    @patch('maya4.dataloader.list_repos_by_author')
    @patch('maya4.dataloader.list_base_files_in_repo')
    @patch('maya4.dataloader.parse_product_filename')
    def test_online_file_discovery(self, mock_parse, mock_list_files, mock_list_repos, temp_dir):
        """Test that remote files are discovered correctly."""
        # Mock repository discovery
        mock_list_repos.return_value = ['Maya4/PT1', 'Maya4/PT2']
        
        # Mock file listing
        def list_files_side_effect(repo_id):
            if 'PT1' in repo_id:
                return ['file1.zarr', 'file2.zarr']
            elif 'PT2' in repo_id:
                return ['file3.zarr']
            return []
        mock_list_files.side_effect = list_files_side_effect
        
        # Mock parsing
        def parse_side_effect(filepath):
            filename = Path(filepath).name
            return {
                'full_name': Path(filepath),
                'part': 'PT1' if 'file1' in filename or 'file2' in filename else 'PT2',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            author='Maya4',
            online=True,
            verbose=False,
            max_products=10
        )
        
        # Verify remote discovery
        assert mock_list_repos.called
        assert mock_list_files.called
        assert len(dataset.get_files()) == 3
    
    @patch('maya4.dataloader.list_repos_by_author')
    @patch('maya4.dataloader.list_base_files_in_repo')
    @patch('maya4.dataloader.parse_product_filename')
    def test_online_with_filtering(self, mock_parse, mock_list_files, mock_list_repos, temp_dir):
        """Test filtering works with online mode."""
        mock_list_repos.return_value = ['Maya4/PT1']
        mock_list_files.return_value = ['file1.zarr', 'file2.zarr', 'file3.zarr']
        
        def parse_side_effect(filepath):
            idx = int(Path(filepath).stem[-1])
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV' if idx % 2 == 1 else 'VH',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        sample_filter = SampleFilter(polarizations=['VV'])
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            filters=sample_filter,
            online=True,
            verbose=False,
            max_products=10
        )
        
        # Should filter to VV only (file1, file3)
        files = dataset.get_files()
        assert len(files) == 2


class TestLazyCoordinateGeneration:
    """Tests for lazy coordinate generation in dataset."""
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_calculate_patches_creates_lazy_coords(self, mock_parse, temp_dir):
        """Test that calculate_patches_from_store creates LazyCoordinateGenerator."""
        # Create a zarr file
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        zfile = part_dir / "test.zarr"
        zfile.mkdir()
        (zfile / "rcmc").mkdir()
        (zfile / "az").mkdir()
        
        # Create zarr arrays
        zarr.open_array(str(zfile / "rcmc"), mode='w', shape=(1000, 1000), chunks=(100, 100), dtype=np.complex64)
        zarr.open_array(str(zfile / "az"), mode='w', shape=(1000, 1000), chunks=(100, 100), dtype=np.complex64)
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False,
            patch_size=(128, 128),
            buffer=(50, 50),
            stride=(64, 64)
        )
        
        # Calculate patches for the file
        files = dataset.get_files()
        if len(files) > 0:
            dataset.calculate_patches_from_store(files[0], patch_order="row")
            
            # Verify lazy coords were created
            samples = dataset.get_samples_by_file(files[0])
            assert samples is not None
            assert isinstance(samples, (list, LazyCoordinateGenerator))
            
            if isinstance(samples, LazyCoordinateGenerator):
                assert len(samples) > 0
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_lazy_coords_iteration(self, mock_parse, temp_dir):
        """Test that lazy coordinates can be iterated."""
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        zfile = part_dir / "test.zarr"
        zfile.mkdir()
        (zfile / "rcmc").mkdir()
        (zfile / "az").mkdir()
        
        zarr.open_array(str(zfile / "rcmc"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
        zarr.open_array(str(zfile / "az"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False,
            patch_size=(64, 64),
            buffer=(32, 32),
            stride=(32, 32)
        )
        
        files = dataset.get_files()
        if len(files) > 0:
            dataset.calculate_patches_from_store(files[0])
            samples = dataset.get_samples_by_file(files[0])
            
            # Try to iterate and get first few coordinates
            coords_list = []
            for i, (y, x) in enumerate(samples):
                coords_list.append((y, x))
                if i >= 4:  # Just get first 5
                    break
            
            assert len(coords_list) == 5
            # Verify they are valid tuples
            for y, x in coords_list:
                assert isinstance(y, (int, np.integer))
                assert isinstance(x, (int, np.integer))


class TestDatasetLength:
    """Tests for dataset length calculation."""
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_len_with_no_files(self, mock_parse, temp_dir):
        """Test __len__ returns 0 with no files."""
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False
        )
        
        assert len(dataset) == 0
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_len_with_lazy_coords(self, mock_parse, temp_dir):
        """Test __len__ counts patches from LazyCoordinateGenerator."""
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        zfile = part_dir / "test.zarr"
        zfile.mkdir()
        (zfile / "rcmc").mkdir()
        (zfile / "az").mkdir()
        
        zarr.open_array(str(zfile / "rcmc"), mode='w', shape=(400, 400), chunks=(100, 100), dtype=np.complex64)
        zarr.open_array(str(zfile / "az"), mode='w', shape=(400, 400), chunks=(100, 100), dtype=np.complex64)
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            online=False,
            verbose=False,
            patch_size=(64, 64),
            buffer=(32, 32),
            stride=(32, 32)
        )
        
        # Calculate patches
        files = dataset.get_files()
        if len(files) > 0:
            dataset.calculate_patches_from_store(files[0])
            
            # Length should be positive
            total_len = len(dataset)
            assert total_len > 0


class TestGetSARDataloader:
    """Tests for get_sar_dataloader factory function."""
    
    @patch('maya4.dataloader.parse_product_filename')
    def test_dataloader_creation_local(self, mock_parse, temp_dir):
        """Test creating dataloader with local files."""
        part_dir = temp_dir / "PT1"
        part_dir.mkdir()
        zfile = part_dir / "test.zarr"
        zfile.mkdir()
        (zfile / "rcmc").mkdir()
        (zfile / "az").mkdir()
        
        zarr.open_array(str(zfile / "rcmc"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
        zarr.open_array(str(zfile / "az"), mode='w', shape=(500, 500), chunks=(100, 100), dtype=np.complex64)
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataloader = get_sar_dataloader(
            data_dir=str(temp_dir),
            online=False,
            batch_size=2,
            num_workers=0,
            verbose=False,
            patch_size=(64, 64),
            max_products=1
        )
        
        assert dataloader is not None
        assert dataloader.dataset is not None
        assert dataloader.batch_size == 2
    
    @patch('maya4.dataloader.list_repos_by_author')
    @patch('maya4.dataloader.list_base_files_in_repo')
    @patch('maya4.dataloader.parse_product_filename')
    def test_dataloader_creation_online(self, mock_parse, mock_list_files, mock_list_repos, temp_dir):
        """Test creating dataloader with online=True."""
        mock_list_repos.return_value = ['Maya4/PT1']
        mock_list_files.return_value = ['file1.zarr']
        
        def parse_side_effect(filepath):
            return {
                'full_name': Path(filepath),
                'part': 'PT1',
                'polarization': 'VV',
                'acquisition_date': '20200101'
            }
        mock_parse.side_effect = parse_side_effect
        
        dataloader = get_sar_dataloader(
            data_dir=str(temp_dir),
            online=True,
            batch_size=4,
            num_workers=0,
            verbose=False,
            max_products=1
        )
        
        assert dataloader is not None
        assert mock_list_repos.called
        assert mock_list_files.called


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
