"""
Tests for utility functions.
"""
from pathlib import Path

import numpy as np
import pytest


class TestSampleFilter:
    """Tests for SampleFilter class."""
    
    def test_initialization(self):
        """Test SampleFilter initialization."""
        from maya4.utils import SampleFilter
        
        filter = SampleFilter(
            parts=['part1', 'part2'],
            years=[2024, 2025],
            polarizations=['HH', 'VV']
        )
        
        assert filter.parts == ['part1', 'part2']
        assert filter.years == [2024, 2025]
        assert filter.polarizations == ['HH', 'VV']


class TestNormalizationFunctions:
    """Tests for normalization utility functions."""
    
    def test_minmax_normalize(self):
        """Test min-max normalization function."""
        from maya4.utils import minmax_normalize
        
        data = np.array([-10.0, 0.0, 10.0])
        normalized = minmax_normalize(data, -10.0, 10.0)
        
        np.testing.assert_array_almost_equal(normalized, [0.0, 0.5, 1.0])
    
    def test_minmax_inverse(self):
        """Test inverse min-max normalization."""
        from maya4.utils import minmax_inverse
        
        normalized = np.array([0.0, 0.5, 1.0])
        denormalized = minmax_inverse(normalized, -10.0, 10.0)
        
        np.testing.assert_array_almost_equal(denormalized, [-10.0, 0.0, 10.0])


class TestFilenameUtilities:
    """Tests for filename parsing utilities."""
    
    def test_extract_stripmap_mode(self):
        """Test extracting stripmap mode from filename."""
        from maya4.utils import extract_stripmap_mode_from_filename
        
        # Real format: s1a-s3-raw-s-hh-...
        filename = "s1a-s3-raw-s-hh-20250101t000000-20250101t000030-057498-0714b4.zarr"
        mode = extract_stripmap_mode_from_filename(filename)
        
        assert mode == 3
    
    def test_extract_stripmap_mode_no_mode(self):
        """Test filename without stripmap mode."""
        from maya4.utils import extract_stripmap_mode_from_filename
        
        filename = "S1A_IW_SLC__1SSH_20250101T000000.zarr"
        mode = extract_stripmap_mode_from_filename(filename)
        
        assert mode is None  # Function returns None, not 0


class TestChunkUtilities:
    """Tests for chunk-related utilities."""
    
    def test_get_chunk_name_from_coords(self):
        """Test generating chunk name from coordinates."""
        from maya4.utils import get_chunk_name_from_coords
        
        chunk_name = get_chunk_name_from_coords(
            y=10, 
            x=20, 
            zarr_file_name="test.zarr",
            level="rcmc"
        )
        
        assert isinstance(chunk_name, str)
        assert 'test.zarr' in chunk_name
        assert 'rcmc' in chunk_name
