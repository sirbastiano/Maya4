"""
Test script for get_balanced_sample_files function.

This script demonstrates how to use the balanced sampling feature
and validates the output distribution.
"""

import os
from pathlib import Path
from maya4.utils import get_balanced_sample_files, SampleFilter

def test_basic_balanced_sampling():
    """Test basic balanced sampling without filters."""
    print("=" * 80)
    print("TEST 1: Basic Balanced Sampling (no filters)")
    print("=" * 80)
    
    data_dir = "/Data/sar_focusing"
    max_samples = 30  # Request 30 samples (10 per scene type ideally)
    
    selected_files = get_balanced_sample_files(
        max_samples=max_samples,
        data_dir=data_dir,
        sample_filter=None,
        split_type="train",
        ensure_representation=True,
        verbose=True,
        repo_author='Maya4',
        repos=['PT1']  # Just PT1 for faster testing
    )
    
    print(f"\n✓ Selected {len(selected_files)} files")
    print(f"\nFirst 5 files:")
    for i, f in enumerate(selected_files[:5], 1):
        print(f"  {i}. {Path(f).name}")
    
    return selected_files


def test_with_polarization_filter():
    """Test balanced sampling with polarization filter."""
    print("\n" + "=" * 80)
    print("TEST 2: Balanced Sampling with Polarization Filter (HH only)")
    print("=" * 80)
    
    data_dir = "/Data/sar_focusing"
    max_samples = 20
    
    # Create filter for HH polarization only
    sample_filter = SampleFilter(polarizations=['hh'])
    
    selected_files = get_balanced_sample_files(
        max_samples=max_samples,
        data_dir=data_dir,
        sample_filter=sample_filter,
        split_type="train",
        ensure_representation=True,
        verbose=True,
        repo_author='Maya4',
        repos=['PT1']
    )
    
    print(f"\n✓ Selected {len(selected_files)} files (all HH polarization)")
    
    # Verify all are HH
    hh_count = sum(1 for f in selected_files if '-hh-' in str(f).lower())
    print(f"  Verification: {hh_count}/{len(selected_files)} are HH polarization")
    
    return selected_files


def test_with_year_filter():
    """Test balanced sampling with year filter."""
    print("\n" + "=" * 80)
    print("TEST 3: Balanced Sampling with Year Filter (2024-2025)")
    print("=" * 80)
    
    data_dir = "/Data/sar_focusing"
    max_samples = 15
    
    # Create filter for specific years
    sample_filter = SampleFilter(years=[2024, 2025])
    
    selected_files = get_balanced_sample_files(
        max_samples=max_samples,
        data_dir=data_dir,
        sample_filter=sample_filter,
        split_type="train",
        ensure_representation=True,
        verbose=True,
        repo_author='Maya4',
        repos=['PT1']
    )
    
    print(f"\n✓ Selected {len(selected_files)} files from 2024-2025")
    
    return selected_files


def test_small_sample():
    """Test with very small sample size."""
    print("\n" + "=" * 80)
    print("TEST 4: Small Sample Size (6 samples - 2 per scene type)")
    print("=" * 80)
    
    data_dir = "/Data/sar_focusing"
    max_samples = 6
    
    selected_files = get_balanced_sample_files(
        max_samples=max_samples,
        data_dir=data_dir,
        sample_filter=None,
        split_type="train",
        ensure_representation=True,
        verbose=True,
        repo_author='Maya4',
        repos=['PT1']
    )
    
    print(f"\n✓ Selected {len(selected_files)} files")
    
    return selected_files


def test_validation_split():
    """Test using validation split instead of train."""
    print("\n" + "=" * 80)
    print("TEST 5: Using Validation Split")
    print("=" * 80)
    
    data_dir = "/Data/sar_focusing"
    max_samples = 10
    
    selected_files = get_balanced_sample_files(
        max_samples=max_samples,
        data_dir=data_dir,
        sample_filter=None,
        split_type="validation",  # Changed to validation
        ensure_representation=True,
        verbose=True,
        repo_author='Maya4',
        repos=['PT1']
    )
    
    print(f"\n✓ Selected {len(selected_files)} files from validation split")
    
    return selected_files


def analyze_distribution(selected_files):
    """Analyze the scene type distribution of selected files."""
    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Count scene types based on filename patterns (rough heuristic)
    land_count = sum(1 for f in selected_files if 'land' in str(f).lower())
    sea_count = sum(1 for f in selected_files if 'sea' in str(f).lower())
    coast_count = sum(1 for f in selected_files if 'coast' in str(f).lower())
    unknown_count = len(selected_files) - (land_count + sea_count + coast_count)
    
    total = len(selected_files)
    
    print(f"\nTotal files: {total}")
    print(f"  Land:    {land_count:3d} ({land_count/total*100:5.1f}%)")
    print(f"  Sea:     {sea_count:3d} ({sea_count/total*100:5.1f}%)")
    print(f"  Coast:   {coast_count:3d} ({coast_count/total*100:5.1f}%)")
    if unknown_count > 0:
        print(f"  Unknown: {unknown_count:3d} ({unknown_count/total*100:5.1f}%)")
    
    print(f"\n  Ideal distribution: 33.3% each")


if __name__ == "__main__":
    print("🧪 Testing Balanced Sampling Function\n")
    
    try:
        # Run tests
        files1 = test_basic_balanced_sampling()
        files2 = test_with_polarization_filter()
        files3 = test_with_year_filter()
        files4 = test_small_sample()
        files5 = test_validation_split()
        
        # Analyze first test results
        analyze_distribution(files1)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
