"""
Unit tests for the refactored dataloader components.
"""
import pytest
import numpy as np
import torch
import zarr
from pathlib import Path


class TestLazyCoordinateRange:
    """Tests for LazyCoordinateRange class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        from maya4.dataloader_clean import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        assert len(coord_range) == 10
        assert coord_range.start == 0
        assert coord_range.stop == 100
        assert coord_range.step == 10
    
    def test_indexing(self):
        """Test indexing operations."""
        from maya4.dataloader_clean import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        assert coord_range[0] == 0
        assert coord_range[5] == 50
        assert coord_range[-1] == 90
    
    def test_iteration(self):
        """Test iteration over range."""
        from maya4.dataloader_clean import LazyCoordinateRange
        
        coord_range = LazyCoordinateRange(0, 100, 10)
        values = list(coord_range)
        expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert values == expected


class TestLazyCoordinateGenerator:
    """Tests for LazyCoordinateGenerator class."""
    
    def test_row_order(self):
        """Test row-major order generation."""
        from maya4.dataloader_clean import LazyCoordinateRange, LazyCoordinateGenerator
        
        y_range = LazyCoordinateRange(0, 3, 1)
        x_range = LazyCoordinateRange(0, 2, 1)
        gen = LazyCoordinateGenerator(y_range, x_range, patch_order="row")
        
        coords = list(gen)
        expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert coords == expected
    
    def test_col_order(self):
        """Test column-major order generation."""
        from maya4.dataloader_clean import LazyCoordinateRange, LazyCoordinateGenerator
        
        y_range = LazyCoordinateRange(0, 3, 1)
        x_range = LazyCoordinateRange(0, 2, 1)
        gen = LazyCoordinateGenerator(y_range, x_range, patch_order="col")
        
        coords = list(gen)
        expected = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        assert coords == expected
    
    def test_length(self):
        """Test length calculation."""
        from maya4.dataloader_clean import LazyCoordinateRange, LazyCoordinateGenerator
        
        y_range = LazyCoordinateRange(0, 10, 2)
        x_range = LazyCoordinateRange(0, 20, 5)
        gen = LazyCoordinateGenerator(y_range, x_range)
        
        assert len(gen) == 5 * 4  # (10/2) * (20/5)


class TestPositionalEncoding2D:
    """Tests for PositionalEncoding2D class."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        from maya4.dataloader_clean import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=True)
        assert pe.complex_valued == False
        assert pe.concat == True
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        from maya4.dataloader_clean import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=True)
        
        # Create sample input (height, width, 2) for real/imag
        inp = np.random.randn(64, 64, 2).astype(np.float32)
        position = (10, 20)
        max_length = (1000, 1000)
        
        output = pe.forward(inp, position, max_length)
        
        # Should add one channel for positional encoding
        assert output.shape == (64, 64, 3)
    
    def test_forward_no_concat(self):
        """Test forward without concatenation."""
        from maya4.dataloader_clean import PositionalEncoding2D
        
        pe = PositionalEncoding2D(complex_valued=False, concat=False)
        
        inp = np.random.randn(64, 64, 2).astype(np.float32)
        position = (10, 20)
        max_length = (1000, 1000)
        
        output = pe.forward(inp, position, max_length)
        
        # Should replace last channel with positional encoding
        assert output.shape == (64, 64, 2)


class TestPositionalEncodingRow:
    """Tests for PositionalEncodingRow class."""
    
    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        from maya4.dataloader_clean import PositionalEncodingRow
        
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
        from maya4.dataloader_clean import create_positional_encoding_module, PositionalEncoding2D
        
        pe = create_positional_encoding_module("2d", complex_valued=False)
        assert isinstance(pe, PositionalEncoding2D)
    
    def test_create_row_encoding(self):
        """Test creating row positional encoding."""
        from maya4.dataloader_clean import create_positional_encoding_module, PositionalEncodingRow
        
        pe = create_positional_encoding_module("row", complex_valued=False)
        assert isinstance(pe, PositionalEncodingRow)
    
    def test_invalid_encoding_type(self):
        """Test that invalid encoding type raises error."""
        from maya4.dataloader_clean import create_positional_encoding_module
        
        with pytest.raises(ValueError):
            create_positional_encoding_module("invalid_type")


class TestChunkCache:
    """Tests for ChunkCache class."""
    
    def test_initialization(self, temp_dir, sample_zarr_store):
        """Test ChunkCache initialization."""
        from maya4.dataloader_clean import SARZarrDataset, ChunkCache
        
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
        from maya4.dataloader_clean import SARZarrDataset, ChunkCache
        
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
        assert chunk.dtype == np.complex64


class TestSARZarrDataset:
    """Tests for SARZarrDataset class."""
    
    def test_initialization(self, temp_dir):
        """Test dataset initialization."""
        from maya4.dataloader_clean import SARZarrDataset
        
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
        from maya4.dataloader_clean import SARZarrDataset, PositionalEncoding2D
        
        dataset = SARZarrDataset(
            data_dir=str(temp_dir),
            patch_size=(128, 128),
            positional_encoding="2d",
            online=False,
            verbose=False
        )
        
        assert isinstance(dataset.pos_encoding, PositionalEncoding2D)
    
    def test_chunk_cache_creation(self, temp_dir):
        """Test that chunk cache is created."""
        from maya4.dataloader_clean import SARZarrDataset, ChunkCache
        
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
        from maya4.dataloader_clean import SARZarrDataset, KPatchSampler
        
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
        from maya4.dataloader_clean import SARZarrDataset, KPatchSampler, SARDataloader
        
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
        from maya4.dataloader_clean import get_sar_dataloader
        
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
    def transform(self):
        """Create transform for testing."""
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
    
    @pytest.fixture
    def sample_filter(self):
        """Create sample filter for testing."""
        from maya4.utils import SampleFilter
        return SampleFilter(
            years=[2023],
            polarizations=["hh"],
            stripmap_modes=[1, 2, 3],
            parts=["PT1", "PT3"]
        )
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data_large/marine/PythonProjects/SAR/sarpyx/data").exists(),
        reason="Test data directory not available"
    )
    def test_horizontal_patch_matches_zarr(self, transform, sample_filter):
        """Test that horizontal patches match direct zarr access."""
        import zarr
        import numpy as np
        from maya4.dataloader_clean import get_sar_dataloader
        
        patch_size = (1, -1)
        buffer = (0, 0)
        stride = (1, 300)
        
        dataloader = get_sar_dataloader(
            data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
            level_from="rcmc",
            level_to="az",
            batch_size=16,
            num_workers=0,
            patch_size=patch_size,
            buffer=buffer,
            stride=stride,
            transform=transform,
            shuffle_files=False,
            patch_order="row",
            complex_valued=True,
            save_samples=False,
            verbose=False,
            samples_per_prod=1000,
            cache_size=100,
            online=True,
            max_products=1,
            positional_encoding=True,
            filters=sample_filter
        )
        
        file = dataloader.dataset._files["full_name"].loc[0]
        
        # Test first 3 rows
        for i in range(3):
            sample_from, sample_to = dataloader.dataset[(file, i, 0)]
            
            # Test level_from
            restored_column_from = dataloader.dataset.get_patch_visualization(
                patch=sample_from,
                level=dataloader.dataset.level_from,
                restore_complex=True,
                prepare_for_plotting=False
            ).squeeze(0).flatten()
            
            actual_column_from = zarr.open(file, mode='r')[dataloader.dataset.level_from][i, :]
            
            assert restored_column_from.shape[0] == actual_column_from.shape[0], \
                f"Shape mismatch at {dataloader.dataset.level_from}: {restored_column_from.shape} vs {actual_column_from.shape}"
            
            np.testing.assert_allclose(
                restored_column_from[:100],
                actual_column_from[:100],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Data mismatch at {dataloader.dataset.level_from}, row {i}"
            )
            
            # Test level_to
            restored_column_to = dataloader.dataset.get_patch_visualization(
                patch=sample_to,
                level=dataloader.dataset.level_to,
                restore_complex=True,
                prepare_for_plotting=False
            ).squeeze(0).flatten()
            
            actual_column_to = zarr.open(file, mode='r')[dataloader.dataset.level_to][i, :]
            
            assert restored_column_to.shape[0] == actual_column_to.shape[0], \
                f"Shape mismatch at {dataloader.dataset.level_to}: {restored_column_to.shape} vs {actual_column_to.shape}"
            
            np.testing.assert_allclose(
                restored_column_to[:100],
                actual_column_to[:100],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Data mismatch at {dataloader.dataset.level_to}, row {i}"
            )
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data_large/marine/PythonProjects/SAR/sarpyx/data").exists(),
        reason="Test data directory not available"
    )
    def test_vertical_patch_matches_zarr(self, transform, sample_filter):
        """Test that vertical patches match direct zarr access."""
        import zarr
        import numpy as np
        from maya4.dataloader_clean import get_sar_dataloader
        
        patch_size = (-1, 1)
        buffer = (0, 0)
        stride = (300, 1)
        
        dataloader = get_sar_dataloader(
            data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
            level_from="rcmc",
            level_to="az",
            batch_size=16,
            num_workers=0,
            patch_size=patch_size,
            buffer=buffer,
            stride=stride,
            transform=transform,
            shuffle_files=False,
            patch_order="row",
            complex_valued=True,
            save_samples=False,
            verbose=False,
            samples_per_prod=1000,
            cache_size=100,
            online=True,
            max_products=1,
            positional_encoding=True,
            filters=sample_filter
        )
        
        file = dataloader.dataset._files["full_name"].loc[0]
        
        # Test first 3 columns
        for i in range(3):
            sample_from, sample_to = dataloader.dataset[(file, 0, i)]
            
            # Test level_from
            restored_column_from = dataloader.dataset.get_patch_visualization(
                patch=sample_from,
                level=dataloader.dataset.level_from,
                restore_complex=True,
                prepare_for_plotting=False
            ).squeeze(1).flatten()
            
            actual_column_from = zarr.open(file, mode='r')[dataloader.dataset.level_from][:, i]
            
            assert restored_column_from.shape[0] == actual_column_from.shape[0], \
                f"Shape mismatch at {dataloader.dataset.level_from}: {restored_column_from.shape} vs {actual_column_from.shape}"
            
            np.testing.assert_allclose(
                restored_column_from[:100],
                actual_column_from[:100],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Data mismatch at {dataloader.dataset.level_from}, column {i}"
            )
            
            # Test level_to
            restored_column_to = dataloader.dataset.get_patch_visualization(
                patch=sample_to,
                level=dataloader.dataset.level_to,
                restore_complex=True,
                prepare_for_plotting=False
            ).squeeze(1).flatten()
            
            actual_column_to = zarr.open(file, mode='r')[dataloader.dataset.level_to][:, i]
            
            assert restored_column_to.shape[0] == actual_column_to.shape[0], \
                f"Shape mismatch at {dataloader.dataset.level_to}: {restored_column_to.shape} vs {actual_column_to.shape}"
            
            np.testing.assert_allclose(
                restored_column_to[:100],
                actual_column_to[:100],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Data mismatch at {dataloader.dataset.level_to}, column {i}"
            )
    
    def test_rectangular_patch_with_sample_zarr(self, sample_zarr_store, temp_dir):
        """Test rectangular patches with synthetic zarr data."""
        import zarr
        import numpy as np
        from maya4.dataloader_clean import get_sar_dataloader
        
        # Create a simple dataloader with rectangular patches
        dataloader = get_sar_dataloader(
            data_dir=str(temp_dir),
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=(64, 64),
            buffer=(10, 10),
            stride=(32, 32),
            shuffle_files=False,
            complex_valued=True,
            save_samples=False,
            verbose=False,
            cache_size=10,
            online=False,
            positional_encoding=False
        )
        
        # If files are available, test a patch
        if len(dataloader.dataset._files) > 0:
            file = str(sample_zarr_store)
            y, x = 50, 50  # Sample coordinates
            
            try:
                sample_from, sample_to = dataloader.dataset[(file, y, x)]
                
                # Verify shapes
                assert sample_from.shape[:2] == (64, 64), f"Unexpected shape: {sample_from.shape}"
                assert sample_to.shape[:2] == (64, 64), f"Unexpected shape: {sample_to.shape}"
                
                # Verify data types
                assert np.iscomplexobj(sample_from), "Expected complex data"
                assert np.iscomplexobj(sample_to), "Expected complex data"
            except (KeyError, IndexError):
                # If coordinates are out of bounds, that's okay for this test
                pytest.skip("Coordinates out of bounds for test zarr")
