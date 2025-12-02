"""
Test script to verify maya4 package installation and imports.
"""

def test_imports():
    """Test that all main components can be imported."""
    print('Testing maya4 imports...')
    
    try:
        # Test main components
        from maya4 import KPatchSampler, SampleFilter, SARDataloader, SARTransform, SARZarrDataset, get_sar_dataloader
        print('✓ Main dataloader components imported successfully')
        
        # Test normalization modules
        from maya4 import BaseTransformModule, ComplexNormalizationModule, IdentityModule, NormalizationModule
        print('✓ Normalization modules imported successfully')
        
        # Test API utilities
        from maya4 import download_metadata_from_product, fetch_chunk_from_hf_zarr, list_base_files_in_repo
        print('✓ API utilities imported successfully')
        
        # Test utility functions
        from maya4 import (
            GT_MAX,
            GT_MIN,
            RC_MAX,
            RC_MIN,
            get_chunk_name_from_coords,
            get_sample_visualization,
            minmax_inverse,
            minmax_normalize,
        )
        print('✓ Utility functions imported successfully')
        
        # Test location utilities
        from maya4 import get_products_spatial_mapping
        print('✓ Location utilities imported successfully')
        
        print('\\n✅ All imports successful!')
        print(f'\\nPackage version: {maya4.__version__}')
        print(f'Package author: {maya4.__author__}')
        
        return True
        
    except ImportError as e:
        print(f'\\n❌ Import failed: {e}')
        return False


def test_basic_functionality():
    """Test basic functionality of key components."""
    print('\\nTesting basic functionality...')
    
    try:
        from maya4 import GT_MAX, GT_MIN, RC_MAX, RC_MIN, SARTransform

        # Create a transform
        transforms = SARTransform.create_minmax_normalized_transform(
            normalize=True,
            rc_min=RC_MIN,
            rc_max=RC_MAX,
            gt_min=GT_MIN,
            gt_max=GT_MAX,
            complex_valued=True
        )
        print('✓ SARTransform created successfully')
        
        # Test normalization
        import numpy as np
        test_data = np.array([1000.0 + 1000.0j], dtype=np.complex128)
        normalized = transforms(test_data, 'rcmc')
        print(f'✓ Normalization test: {test_data[0]} -> {normalized[0]}')
        
        print('\\n✅ Basic functionality tests passed!')
        return True
        
    except Exception as e:
        print(f'\\n❌ Functionality test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import maya4
    
    print('='*60)
    print('MAYA4 PACKAGE INSTALLATION TEST')
    print('='*60)
    print()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality if imports succeeded
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    print()
    print('='*60)
    if imports_ok and functionality_ok:
        print('✅ ALL TESTS PASSED')
        print('The maya4 package is installed and working correctly!')
    else:
        print('❌ SOME TESTS FAILED')
        print('Please check the error messages above.')
    print('='*60)
