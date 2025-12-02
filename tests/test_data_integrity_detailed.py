"""
Detailed data integrity tests with difference reporting and inverse transform validation.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr


class TestDetailedDataIntegrity:
    """Detailed tests that show exact differences and validate inverse transforms."""
    
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
    def test_raw_data_with_difference_report(self, sample_filter):
        """Test raw data loading and report exact differences."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader
        
        patch_size = (64, 128)
        
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=(0, 0),
            stride=(64, 128),
            transform=None,  # No transformation
            shuffle_files=False,
            complex_valued=True,
            verbose=False,
            cache_size=10,
            online=False,
            max_products=1,
            positional_encoding=False,
            filters=sample_filter,
            use_balanced_sampling=False
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset._files["full_name"].iloc[0])
        store = zarr.open(file, mode='r')
        
        # Test a patch at origin
        y, x = 0, 0
        patch_from, patch_to = dataloader.dataset[(file, y, x)]
        
        # Convert to numpy
        patch_from_np = patch_from.numpy()
        patch_to_np = patch_to.numpy()
        
        # Get zarr data
        actual_from = store['rcmc'][y:y+patch_size[0], x:x+patch_size[1]]
        actual_to = store['az'][y:y+patch_size[0], x:x+patch_size[1]]
        
        # Calculate differences
        diff_from = np.abs(patch_from_np - actual_from)
        diff_to = np.abs(patch_to_np - actual_to)
        
        print(f"\n{'='*80}")
        print(f"RAW DATA COMPARISON REPORT")
        print(f"{'='*80}")
        print(f"File: {Path(file).name}")
        print(f"Patch location: ({y}, {x})")
        print(f"Patch size: {patch_size}")
        print(f"\nRCMC Level:")
        print(f"  Dataloader shape: {patch_from_np.shape}")
        print(f"  Zarr shape: {actual_from.shape}")
        print(f"  Max absolute difference: {diff_from.max():.2e}")
        print(f"  Mean absolute difference: {diff_from.mean():.2e}")
        print(f"  Std absolute difference: {diff_from.std():.2e}")
        print(f"  Number of exact matches: {np.sum(diff_from == 0)} / {diff_from.size}")
        
        print(f"\nAZ Level:")
        print(f"  Dataloader shape: {patch_to_np.shape}")
        print(f"  Zarr shape: {actual_to.shape}")
        print(f"  Max absolute difference: {diff_to.max():.2e}")
        print(f"  Mean absolute difference: {diff_to.mean():.2e}")
        print(f"  Std absolute difference: {diff_to.std():.2e}")
        print(f"  Number of exact matches: {np.sum(diff_to == 0)} / {diff_to.size}")
        
        if diff_from.max() > 0:
            print(f"\nRCMC Sample values (first 3x3):")
            print(f"  Dataloader: {patch_from_np[:3, :3]}")
            print(f"  Zarr:       {actual_from[:3, :3]}")
        
        # Assert they match
        np.testing.assert_allclose(
            patch_from_np, actual_from,
            rtol=1e-6, atol=1e-8,
            err_msg="RCMC data mismatch"
        )
        np.testing.assert_allclose(
            patch_to_np, actual_to,
            rtol=1e-6, atol=1e-8,
            err_msg="AZ data mismatch"
        )
        
        print(f"\n✓ All data matches within tolerance!")
        print(f"{'='*80}\n")
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data/sar_focusing").exists(),
        reason="Test data directory not available"
    )
    def test_with_normalization_and_inverse(self, sample_filter):
        """Test that normalization and its inverse properly round-trip the data."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader
        from maya4.normalization import SARTransform
        from maya4.utils import GT_MAX, GT_MIN, RC_MAX, RC_MIN

        # Create normalization transform
        transform = SARTransform.create_minmax_normalized_transform(
            normalize=True,
            rc_min=RC_MIN,
            rc_max=RC_MAX,
            gt_min=GT_MIN,
            gt_max=GT_MAX,
            complex_valued=True
        )
        
        patch_size = (64, 128)
        
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=(0, 0),
            stride=(64, 128),
            transform=transform,  # Apply normalization
            shuffle_files=False,
            complex_valued=True,
            verbose=False,
            cache_size=10,
            online=False,
            max_products=1,
            positional_encoding=False,
            filters=sample_filter,
            use_balanced_sampling=False
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset._files["full_name"].iloc[0])
        store = zarr.open(file, mode='r')
        
        # Test a patch
        y, x = 0, 0
        
        # Get normalized data from dataloader
        patch_from_norm, patch_to_norm = dataloader.dataset[(file, y, x)]
        patch_from_norm_np = patch_from_norm.numpy()
        patch_to_norm_np = patch_to_norm.numpy()
        
        # Get raw zarr data
        actual_from = store['rcmc'][y:y+patch_size[0], x:x+patch_size[1]]
        actual_to = store['az'][y:y+patch_size[0], x:x+patch_size[1]]
        
        # Apply transform manually to raw data
        expected_from_norm = transform(actual_from, 'rcmc')
        expected_to_norm = transform(actual_to, 'az')
        
        # Apply inverse transform to get back original data
        restored_from = transform.inverse(patch_from_norm_np, 'rcmc')
        restored_to = transform.inverse(patch_to_norm_np, 'az')
        
        print(f"\n{'='*80}")
        print(f"NORMALIZATION AND INVERSE TRANSFORM VALIDATION")
        print(f"{'='*80}")
        print(f"File: {Path(file).name}")
        print(f"Patch location: ({y}, {x})")
        print(f"Patch size: {patch_size}")
        
        # Test forward transform consistency
        diff_forward_from = np.abs(patch_from_norm_np - expected_from_norm)
        diff_forward_to = np.abs(patch_to_norm_np - expected_to_norm)
        
        print(f"\nFORWARD TRANSFORM (dataloader vs manual transform):")
        print(f"  RCMC - Max diff: {diff_forward_from.max():.2e}, Mean diff: {diff_forward_from.mean():.2e}")
        print(f"  AZ   - Max diff: {diff_forward_to.max():.2e}, Mean diff: {diff_forward_to.mean():.2e}")
        
        # Test inverse transform (round-trip)
        diff_inverse_from = np.abs(restored_from - actual_from)
        diff_inverse_to = np.abs(restored_to - actual_to)
        
        print(f"\nINVERSE TRANSFORM (restored vs original):")
        print(f"  RCMC - Max diff: {diff_inverse_from.max():.2e}, Mean diff: {diff_inverse_from.mean():.2e}")
        print(f"  AZ   - Max diff: {diff_inverse_to.max():.2e}, Mean diff: {diff_inverse_to.mean():.2e}")
        
        # Show sample values
        print(f"\nSample values (position [0,0]):")
        print(f"  Original RCMC:    {actual_from[0, 0]}")
        print(f"  Normalized:       {patch_from_norm_np[0, 0]}")
        print(f"  Restored:         {restored_from[0, 0]}")
        print(f"  Round-trip error: {np.abs(restored_from[0, 0] - actual_from[0, 0]):.2e}")
        
        # Check normalized values are in expected range
        print(f"\nNormalized value ranges:")
        print(f"  RCMC - min: {patch_from_norm_np.min():.4f}, max: {patch_from_norm_np.max():.4f}")
        print(f"  AZ   - min: {patch_to_norm_np.min():.4f}, max: {patch_to_norm_np.max():.4f}")
        
        # Assert forward transform matches
        np.testing.assert_allclose(
            patch_from_norm_np, expected_from_norm,
            rtol=1e-6, atol=1e-8,
            err_msg="Forward transform mismatch for RCMC"
        )
        np.testing.assert_allclose(
            patch_to_norm_np, expected_to_norm,
            rtol=1e-6, atol=1e-8,
            err_msg="Forward transform mismatch for AZ"
        )
        
        # Assert inverse transform recovers original
        np.testing.assert_allclose(
            restored_from, actual_from,
            rtol=1e-5, atol=1e-6,
            err_msg="Inverse transform failed to recover RCMC data"
        )
        np.testing.assert_allclose(
            restored_to, actual_to,
            rtol=1e-5, atol=1e-6,
            err_msg="Inverse transform failed to recover AZ data"
        )
        
        print(f"\n✓ Forward and inverse transforms validated successfully!")
        print(f"{'='*80}\n")
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("/Data/sar_focusing").exists(),
        reason="Test data directory not available"
    )
    def test_with_positional_encoding_and_removal(self, sample_filter):
        """Test that positional encoding can be added and removed correctly."""
        import numpy as np
        import zarr

        from maya4.dataloader import get_sar_dataloader
        
        patch_size = (64, 128)
        
        # Create dataloader WITH positional encoding
        dataloader = get_sar_dataloader(
            data_dir="/Data/sar_focusing",
            level_from="rcmc",
            level_to="az",
            batch_size=1,
            num_workers=0,
            patch_size=patch_size,
            buffer=(0, 0),
            stride=(64, 128),
            transform=None,
            shuffle_files=False,
            complex_valued=False,  # Split to real/imag
            verbose=False,
            cache_size=10,
            online=False,
            max_products=1,
            positional_encoding="2d",  # Enable 2D positional encoding
            filters=sample_filter,
            use_balanced_sampling=False
        )
        
        if len(dataloader.dataset.get_files()) == 0:
            pytest.skip("No files found matching filter criteria")
        
        file = str(dataloader.dataset._files["full_name"].iloc[0])
        store = zarr.open(file, mode='r')
        
        # Get patch with positional encoding
        y, x = 10, 20
        patch_from_pe, patch_to_pe = dataloader.dataset[(file, y, x)]
        patch_from_pe_np = patch_from_pe.numpy()
        patch_to_pe_np = patch_to_pe.numpy()
        
        # Get raw zarr data
        actual_from = store['rcmc'][y:y+patch_size[0], x:x+patch_size[1]]
        actual_to = store['az'][y:y+patch_size[0], x:x+patch_size[1]]
        
        # Convert to real/imag stacked format (what dataloader does before PE)
        actual_from_stacked = np.stack([np.real(actual_from), np.imag(actual_from)], axis=-1).astype(np.float32)
        actual_to_stacked = np.stack([np.real(actual_to), np.imag(actual_to)], axis=-1).astype(np.float32)
        
        print(f"\n{'='*80}")
        print(f"POSITIONAL ENCODING VALIDATION")
        print(f"{'='*80}")
        print(f"File: {Path(file).name}")
        print(f"Patch location: ({y}, {x})")
        print(f"Patch size: {patch_size}")
        
        print(f"\nData shapes:")
        print(f"  Original zarr shape (RCMC): {actual_from.shape}")
        print(f"  Stacked real/imag shape: {actual_from_stacked.shape}")
        print(f"  With positional encoding: {patch_from_pe_np.shape}")
        
        # Extract original data (first 2 channels are real/imag)
        extracted_from = patch_from_pe_np[:, :, :2]
        extracted_to = patch_to_pe_np[:, :, :2]
        
        # Extract positional encoding (remaining channels)
        pe_channels_from = patch_from_pe_np[:, :, 2:]
        pe_channels_to = patch_to_pe_np[:, :, 2:]
        
        print(f"\nPositional encoding details:")
        print(f"  Number of PE channels added: {pe_channels_from.shape[-1]}")
        print(f"  PE channel values (sample at [0,0]): {pe_channels_from[0, 0, :]}")
        
        # Compare extracted data with original
        diff_from = np.abs(extracted_from - actual_from_stacked)
        diff_to = np.abs(extracted_to - actual_to_stacked)
        
        print(f"\nExtracted data vs original (after removing PE):")
        print(f"  RCMC - Max diff: {diff_from.max():.2e}, Mean diff: {diff_from.mean():.2e}")
        print(f"  AZ   - Max diff: {diff_to.max():.2e}, Mean diff: {diff_to.mean():.2e}")
        
        # Verify positional encoding is non-zero
        assert pe_channels_from.max() > 0, "Positional encoding appears to be zero"
        assert pe_channels_to.max() > 0, "Positional encoding appears to be zero"
        
        # Verify original data is preserved in first 2 channels
        np.testing.assert_allclose(
            extracted_from, actual_from_stacked,
            rtol=1e-6, atol=1e-8,
            err_msg="Original data not preserved after positional encoding"
        )
        np.testing.assert_allclose(
            extracted_to, actual_to_stacked,
            rtol=1e-6, atol=1e-8,
            err_msg="Original data not preserved after positional encoding"
        )
        
        print(f"\n✓ Positional encoding validated - original data preserved in first 2 channels!")
        print(f"{'='*80}\n")
