"""
Tests for normalization modules.
"""
import numpy as np
import pytest


class TestNormalizationModule:
    """Tests for NormalizationModule class."""
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        from maya4.normalization import NormalizationModule
        
        module = NormalizationModule(
            normalize=True,
            min_val=-10.0,
            max_val=10.0
        )
        
        # Test data in range
        data = np.array([[-10.0, 0.0, 10.0]])
        normalized = module.forward(data)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        np.testing.assert_array_almost_equal(normalized, [[0.0, 0.5, 1.0]])
    
    def test_inverse_normalization(self):
        """Test inverse normalization."""
        from maya4.normalization import NormalizationModule
        
        module = NormalizationModule(
            normalize=True,
            min_val=-10.0,
            max_val=10.0
        )
        
        normalized = np.array([[0.0, 0.5, 1.0]])
        denormalized = module.inverse(normalized)
        
        np.testing.assert_array_almost_equal(denormalized, [[-10.0, 0.0, 10.0]])


class TestComplexNormalizationModule:
    """Tests for ComplexNormalizationModule class."""
    
    def test_complex_normalization(self):
        """Test complex-valued normalization."""
        from maya4.normalization import ComplexNormalizationModule
        
        module = ComplexNormalizationModule(
            normalize=True,
            min_val=-10.0,
            max_val=10.0
        )
        
        # Create complex data
        data = np.array([[-10.0 + 5.0j, 0.0 + 0.0j, 10.0 - 5.0j]])
        normalized = module.forward(data)
        
        assert normalized.dtype == np.complex64 or normalized.dtype == np.complex128


class TestSARTransform:
    """Tests for SARTransform class."""
    
    def test_create_normalized_transform(self):
        """Test creating a normalized transform."""
        from maya4.normalization import SARTransform
        
        transform = SARTransform.create_minmax_normalized_transform(
            normalize=True,
            rc_min=-100.0,
            rc_max=100.0,
            gt_min=-50.0,
            gt_max=50.0,
            complex_valued=True
        )
        
        assert transform is not None
        assert 'rc' in transform.modules
        assert 'az' in transform.modules
    
    def test_transform_application(self):
        """Test applying transform to data."""
        from maya4.normalization import SARTransform
        
        transform = SARTransform.create_minmax_normalized_transform(
            normalize=True,
            rc_min=-100.0,
            rc_max=100.0,
            gt_min=-50.0,
            gt_max=50.0,
            complex_valued=True
        )
        
        # Create sample data
        data = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
        transformed = transform(data, 'rc')
        
        assert transformed.shape == data.shape
