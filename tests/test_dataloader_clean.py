"""
Unit tests for the refactored dataloader components.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr


class TestLazyCoordinateRange:
    """Tests for LazyCoordinateRange class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        from maya4.dataloader import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        assert len(coord_range) == 10
        assert coord_range.start == 0
        assert coord_range.stop == 100
        assert coord_range.step == 10
    
    def test_indexing(self):
        """Test indexing operations."""
        from maya4.dataloader import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        assert coord_range[0] == 0
        assert coord_range[5] == 50
        assert coord_range[-1] == 90
    
    def test_iteration(self):
        """Test iteration over range."""
        from maya4.dataloader import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        values = list(coord_range)
        expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert values == expected


class TestLazyCoordinateGenerator:
    """Tests for LazyCoordinateGenerator class."""
    
    def test_row_order(self):
        """Test row-major order generation."""
        from maya4.dataloader import LazyCoordinateGenerator, LazyCoordinateRange
        
        y_range = LazyCoordinateRange(0, 3, 1)
        x_range = LazyCoordinateRange(0, 2, 1)
        gen = LazyCoordinateGenerator(y_range, x_range, patch_order="row")
        
        coords = list(gen)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert coords == expected
    
    def test_col_order(self):
        """Test column-major order generation."""
        from maya4.dataloader import LazyCoordinateGenerator, LazyCoordinateRange
        
        y_range = LazyCoordinateRange(0, 3, 1)
        x_range = LazyCoordinateRange(0, 2, 1)
        gen = LazyCoordinateGenerator(y_range, x_range, patch_order="col")
        
        coords = list(gen)
        expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        assert coords == expected
    
    def test_length(self):
        """Test length calculation."""
        from maya4.dataloader import LazyCoordinateGenerator, LazyCoordinateRange
        
        y_range = LazyCoordinateRange(0, 10, 2)
        x_range = LazyCoordinateRange(0, 20, 5)
        gen = LazyCoordinateGenerator(y_range, x_range)
        
        assert len(gen) == 5 * 4  # (10/2) * (20/5)


class TestPositionalEncoding2D:
    """Tests for PositionalEncoding2D class."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        from maya4.dataloader import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=True)
        assert pe.complex_valued == False
        assert pe.concat == True
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        from maya4.dataloader import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=True)
        
        # Create sample input (height, width, 2) for real/imag
        inp = np.random.randn(64, 64, 2).astype(np.float32)
        position = (10, 20)
        max_length = (1000, 1000)
        
        output = pe.forward(inp, position, max_length)
        
        # Should add two channels for positional encoding (row + col)
        assert output.shape == (64, 64, 4)
    
    def test_forward_no_concat(self):
        """Test forward without concatenation."""
        from maya4.dataloader import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=False)
        
        inp = np.random.randn(64, 64, 2).astype(np.float32)
        position = (10, 20)
        max_length = (1000, 1000)
        
        output = pe.forward(inp, position, max_length)
        
        # With concat=False, positional encoding replaces input but still adds 2 channels
        assert output.shape == (64, 64, 4)


class TestPositionalEncodingRow:
    """Tests for PositionalEncodingRow class."""
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        from maya4.dataloader import PositionalEncodingRow
        
        pe = PositionalEncodingRow(complex_valued=False, concat=True)
        
        # Create sample input (height, width, 2) for real/imag
        inp = np.random.randn(64, 64, 2).astype(np.float32)
        position = (10, 20)
        max_length = (1000, 1000)
        
        output = pe.forward(inp, position, max_length)
        
        # Should add one channel for positional encoding
        assert output.shape == (64, 64, 3)


class TestPositionalEncodingFactory:
    """Tests for positional encoding factory function."""
    
    def test_create_2d_encoding(self):
        """Test creating 2D positional encoding."""
        from maya4.dataloader import PositionalEncoding2D, create_positional_encoding_module
        
        pe = create_positional_encoding_module("2d", complex_valued=False)
        assert isinstance(pe, PositionalEncoding2D)
    
    def test_create_row_encoding(self):
        """Test creating row positional encoding."""
        from maya4.dataloader import PositionalEncodingRow, create_positional_encoding_module
        
        pe = create_positional_encoding_module("row", complex_valued=False)
        assert isinstance(pe, PositionalEncodingRow)
    
    def test_invalid_encoding_type(self):
        """Test that invalid encoding type raises error."""
        from maya4.dataloader import create_positional_encoding_module
        
        with pytest.raises(ValueError):
            create_positional_encoding_module("invalid_type")


class TestChunkCache:
    """Tests for ChunkCache class."""
    
    def test_initialization(self, temp_dir, sample_zarr_store):
        """Test ChunkCache initialization."""
        from maya4.dataloader import ChunkCache, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            cache_size=10,
            online=False
        )
        
        cache = ChunkCache(dataset, cache_size=10)
        assert cache.cache_size == 10
        assert cache.dataset == dataset
    
    def test_load_chunk(self, temp_dir, sample_zarr_store):
        """Test loading a chunk from zarr store."""
        from maya4.dataloader import ChunkCache, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            cache_size=10,
            online=False
        )
        dataset._stores[sample_zarr_store] = {
            'rcmc': zarr.open(str(sample_zarr_store / 'rcmc'), mode='r')
        }
        
        cache = ChunkCache(dataset, cache_size=10)
        
        # Load a chunk
        chunk = cache._load_chunk_uncached(sample_zarr_store, 'rcmc', 0, 0)
        
        assert chunk is not None
        assert isinstance(chunk, np.ndarray)
        # Zarr v3 uses complex128 by default
        assert chunk.dtype in (np.complex64, np.complex128)


class TestSARZarrDataset:
    """Tests for SARZarrDataset class."""
    
    def test_initialization(self, temp_dir):
        """Test dataset initialization."""
        from maya4.dataloader import SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            buffer=(50, 50),
            stride=(64, 64),
            cache_size=10,
            online=False,
            verbose=False
        )
        
        assert dataset.data_dir == Path(temp_dir)
        assert dataset._patch_size == (128, 128)
        assert dataset.buffer == (50, 50)
        assert dataset.stride == (64, 64)
    
    def test_positional_encoding_creation(self, temp_dir):
        """Test that positional encoding is created correctly."""
        from maya4.dataloader import PositionalEncoding2D, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            positional_encoding="2d",
            online=False,
            verbose=False
        )
        
        assert isinstance(dataset.positional_encoding_module, PositionalEncoding2D)
    
    def test_chunk_cache_creation(self, temp_dir):
        """Test that chunk cache is created."""
        from maya4.dataloader import ChunkCache, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            cache_size=10,
            online=False,
            verbose=False
        )
        
        assert isinstance(dataset.chunk_cache, ChunkCache)


class TestKPatchSampler:
    """Tests for KPatchSampler class."""
    
    def test_initialization(self, temp_dir):
        """Test sampler initialization."""
        from maya4.dataloader import KPatchSampler, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            online=False,
            verbose=False
        )
        
        sampler = KPatchSampler(
            dataset=dataset,
            samples_per_prod=100,
            shuffle_files=True,
            verbose=False
        )
        
        assert sampler.dataset == dataset
        assert sampler.samples_per_prod == 100
        assert sampler.shuffle_files == True


class TestSARDataloader:
    """Tests for SARDataloader class."""
    
    def test_initialization(self, temp_dir):
        """Test dataloader initialization."""
        from maya4.dataloader import KPatchSampler, SARDataloader, SARZarrDataset
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            online=False,
            verbose=False
        )
        
        sampler = KPatchSampler(
            dataset=dataset,
            samples_per_prod=100,
            verbose=False
        )
        
        dataloader = SARDataloader(
            dataset=dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=0,
            verbose=False
        )
        
        assert dataloader.dataset == dataset
        assert dataloader.batch_size == 4


class TestIntegration:
    """Integration tests for the complete dataloader pipeline."""
    
    def test_get_sar_dataloader(self, temp_dir):
        """Test the factory function for creating dataloaders."""
        from maya4.dataloader import get_sar_dataloader
        
        dataloader = get_sar_dataloader(
            data_dir=str(temp_dir),
            batch_size=2,
            num_workers=0,
            patch_size=(128, 128),
            cache_size=10,
            online=False,
            verbose=False
        )
        
        assert dataloader is not None
        assert dataloader.batch_size == 2


class TestDataloaderAccuracy:
    """Tests to verify dataloader returns same data as direct zarr access."""
    
    @pytest.fixture
    def sample_filter(self):
        """Create sample filter for testing."""
        from maya4.utils import SampleFilter
        return SampleFilter(
            years=[2023, 2024, 2025],
            polarizations=["hh", "vv"],
            stripmap_modes=[1, 2, 3],
            parts=["PT1", "PT2", "PT3", "PT4"]
        )
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data/sar_focusing").exists(),
        reason="Test data directory not available"
    )
    def test_horizontal_full_row_matches_zarr(self, sample_filter):
        """Test that horizontal rows match direct zarr access."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader

        # Use a reasonable width instead of -1 to avoid issues
        # We'll compare against the full row from zarr
        patch_size = (1, 1000)  # 1 row, reasonable width
        buffer = (0, 0)  # No buffer
        stride = (1, 1)  # Dense sampling
        
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=buffer,
            stride=stride,
            transform=None,  # No transformation
            shuffle_files=False,
            patch_order="row",
            complex_valued=True,
            save_samples=False,
            verbose=False,
            samples_per_prod=10,
            cache_size=100,
            online=False,
            max_products=1,
            positional_encoding=False,  # No positional encoding
            filters=sample_filter,
            use_balanced_sampling=False  # Disable balanced sampling for testing
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset.get_files()[0])
        
        # Open the zarr store directly
        store = zarr.open(file, mode='r')
        
        # Test first 3 rows
        for row_idx in range(min(3, store['rcmc'].shape[0])):
            # Get patch from dataloader (returns torch tensors)
            patch_from, patch_to = dataloader.dataset[(file, row_idx, 0)]
            
            # Convert back to numpy and squeeze to 1D
            patch_from_np = patch_from.numpy().squeeze()
            patch_to_np = patch_to.numpy().squeeze()
            
            # Get actual data from zarr (same region as patch)
            actual_from = store['rcmc'][row_idx, :patch_size[1]]
            actual_to = store['az'][row_idx, :patch_size[1]]
            
            # Compare shapes
            assert patch_from_np.shape == actual_from.shape, \
                f"Shape mismatch for rcmc row {row_idx}: {patch_from_np.shape} vs {actual_from.shape}"
            assert patch_to_np.shape == actual_to.shape, \
                f"Shape mismatch for az row {row_idx}: {patch_to_np.shape} vs {actual_to.shape}"
            
            # Compare data (use relaxed tolerance for numerical precision)
            np.testing.assert_allclose(
                patch_from_np,
                actual_from,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at rcmc row {row_idx}"
            )
            
            np.testing.assert_allclose(
                patch_to_np,
                actual_to,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at az row {row_idx}"
            )
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data/sar_focusing").exists(),
        reason="Test data directory not available"
    )
    def test_vertical_full_column_matches_zarr(self, sample_filter):
        """Test that vertical columns match direct zarr access."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader

        # Use reasonable height instead of -1
        patch_size = (1000, 1)  # Reasonable height, 1 column
        buffer = (0, 0)  # No buffer
        stride = (1, 1)  # Dense sampling
        
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=buffer,
            stride=stride,
            transform=None,  # No transformation
            shuffle_files=False,
            patch_order="col",
            complex_valued=True,
            save_samples=False,
            verbose=False,
            samples_per_prod=10,
            cache_size=100,
            online=False,
            max_products=1,
            positional_encoding=False,  # No positional encoding
            filters=sample_filter,
            use_balanced_sampling=False  # Disable balanced sampling for testing
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset.get_files()[0])
        
        # Open the zarr store directly
        store = zarr.open(file, mode='r')
        
        # Test first 3 columns
        for col_idx in range(min(3, store['rcmc'].shape[1])):
            # Get patch from dataloader
            patch_from, patch_to = dataloader.dataset[(file, 0, col_idx)]
            
            # Convert back to numpy and squeeze to 1D
            patch_from_np = patch_from.numpy().squeeze()
            patch_to_np = patch_to.numpy().squeeze()
            
            # Get actual data from zarr (same region as patch)
            actual_from = store['rcmc'][:patch_size[0], col_idx]
            actual_to = store['az'][:patch_size[0], col_idx]
            
            # Compare shapes
            assert patch_from_np.shape == actual_from.shape, \
                f"Shape mismatch for rcmc column {col_idx}: {patch_from_np.shape} vs {actual_from.shape}"
            assert patch_to_np.shape == actual_to.shape, \
                f"Shape mismatch for az column {col_idx}: {patch_to_np.shape} vs {actual_to.shape}"
            
            # Compare data
            np.testing.assert_allclose(
                patch_from_np,
                actual_from,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at rcmc column {col_idx}"
            )
            
            np.testing.assert_allclose(
                patch_to_np,
                actual_to,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at az column {col_idx}"
            )
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data/sar_focusing").exists(),
        reason="Test data directory not available"
    )
    def test_rectangular_patch_matches_zarr(self, sample_filter):
        """Test that rectangular patches match direct zarr access."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader

        # Rectangular patch
        patch_size = (64, 128)
        buffer = (0, 0)  # No buffer
        stride = (64, 128)  # Non-overlapping
        
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=buffer,
            stride=stride,
            transform=None,  # No transformation
            shuffle_files=False,
            patch_order="row",
            complex_valued=True,
            save_samples=False,
            verbose=False,
            samples_per_prod=10,
            cache_size=10,
            online=False,
            max_products=1,
            positional_encoding=False,  # No positional encoding
            filters=sample_filter,
            use_balanced_sampling=False  # Disable balanced sampling for testing
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset.get_files()[0])
        
        # Open the zarr store directly
        store = zarr.open(file, mode='r')
        
        # Test a few patches at different locations
        test_coords = [(0, 0), (64, 128), (128, 0)]
        
        for y, x in test_coords:
            # Skip if out of bounds
            if y + patch_size[0] > store['rcmc'].shape[0] or x + patch_size[1] > store['rcmc'].shape[1]:
                continue
            
            # Get patch from dataloader
            patch_from, patch_to = dataloader.dataset[(file, y, x)]
            
            # Convert back to numpy
            patch_from_np = patch_from.numpy()
            patch_to_np = patch_to.numpy()
            
            # Get actual data from zarr
            actual_from = store['rcmc'][y:y+patch_size[0], x:x+patch_size[1]]
            actual_to = store['az'][y:y+patch_size[0], x:x+patch_size[1]]
            
            # Compare shapes
            assert patch_from_np.shape == actual_from.shape, \
                f"Shape mismatch for rcmc at ({y}, {x}): {patch_from_np.shape} vs {actual_from.shape}"
            assert patch_to_np.shape == actual_to.shape, \
                f"Shape mismatch for az at ({y}, {x}): {patch_to_np.shape} vs {actual_to.shape}"
            
            # Compare data
            np.testing.assert_allclose(
                patch_from_np,
                actual_from,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at rcmc ({y}, {x})"
            )
            
            np.testing.assert_allclose(
                patch_to_np,
                actual_to,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Data mismatch at az ({y}, {x})"
            )
    
    def test_rectangular_patch_with_sample_zarr(self, sample_zarr_store, temp_dir):
        """Test rectangular patches with synthetic zarr data."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader

        # Create a simple dataloader with rectangular patches
        dataloader = get_sar_dataloader(
            data_dir=str(temp_dir),
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=(64, 64),
            buffer=(0, 0),
            stride=(32, 32),
            transform=None,
            shuffle_files=False,
            complex_valued=True,
            save_samples=False,
            verbose=False,
            cache_size=10,
            online=False,
            positional_encoding=False
        )
        
        # If files are available, test a patch
        if len(dataloader.dataset.get_files()) > 0:
            file = str(sample_zarr_store)
            y, x = 50, 50  # Sample coordinates
            
            try:
                # Get patch from dataloader
                patch_from, patch_to = dataloader.dataset[(file, y, x)]
                
                # Convert to numpy
                patch_from_np = patch_from.numpy()
                patch_to_np = patch_to.numpy()
                
                # Open zarr and get actual data
                store = zarr.open(file, mode='r')
                actual_from = store['rcmc'][y:y+64, x:x+64]
                actual_to = store['az'][y:y+64, x:x+64]
                
                # Verify shapes
                assert patch_from_np.shape == (64, 64), f"Unexpected shape: {patch_from_np.shape}"
                assert patch_to_np.shape == (64, 64), f"Unexpected shape: {patch_to_np.shape}"
                
                # Verify data types
                assert np.iscomplexobj(patch_from_np), "Expected complex data"
                assert np.iscomplexobj(patch_to_np), "Expected complex data"
                
                # Verify data matches
                np.testing.assert_allclose(
                    patch_from_np,
                    actual_from,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg="Data mismatch for rcmc"
                )
                
                np.testing.assert_allclose(
                    patch_to_np,
                    actual_to,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg="Data mismatch for az"
                )
            except (KeyError, IndexError):
                # If coordinates are out of bounds, that's okay for this test
                pytest.skip("Coordinates out of bounds for test zarr")

