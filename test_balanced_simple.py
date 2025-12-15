"""
Simple test to understand get_balanced_sample_files function behavior.
"""

from maya4.utils import get_balanced_sample_files, SampleFilter

def simple_test():
    """
    Simple test showing how get_balanced_sample_files works.
    
    This function:
    1. Loads pre-computed dataset splits (train/val/test)
    2. Applies optional filters (year, month, polarization, etc.)
    3. Selects samples ensuring:
       - Equal scene type distribution (land/sea/coast = 33.3% each)
       - No geographic overlap between samples
       - Respects the train/val/test split
    """
    
    print("📊 Understanding get_balanced_sample_files()\n")
    print("=" * 70)
    
    # Example 1: Get 12 balanced samples (4 per scene type ideally)
    print("\n1️⃣  Get 12 balanced samples from train split:")
    print("-" * 70)
    
    files = get_balanced_sample_files(
        max_samples=12,
        data_dir="/Data/sar_focusing",
        sample_filter=None,  # No filter = all products considered
        split_type="train",  # Use training split
        ensure_representation=True,  # Enable balanced sampling
        verbose=True,  # Show detailed output
        repos=['PT1']  # Just PT1 for speed
    )
    
    print(f"\n✓ Got {len(files)} files")
    
    # Example 2: With filtering
    print("\n" + "=" * 70)
    print("\n2️⃣  Get 6 balanced samples with HH polarization filter:")
    print("-" * 70)
    
    filter_hh = SampleFilter(polarizations=['hh'])
    
    files_hh = get_balanced_sample_files(
        max_samples=6,
        data_dir="/Data/sar_focusing",
        sample_filter=filter_hh,  # Only HH polarization
        split_type="train",
        ensure_representation=True,
        verbose=True,
        repos=['PT1']
    )
    
    print(f"\n✓ Got {len(files_hh)} HH files")
    
    # Show what we got
    print("\n" + "=" * 70)
    print("\n📋 Sample Output:")
    print("-" * 70)
    for i, f in enumerate(files[:3], 1):
        from pathlib import Path
        print(f"{i}. {Path(f).name}")
    
    print("\n" + "=" * 70)
    print("\n💡 KEY POINTS:")
    print("-" * 70)
    print("• The function aims for EQUAL scene type distribution (33.3% each)")
    print("• Uses geographic polygons to avoid overlapping samples")
    print("• Round-robin filling ensures fair representation when one type lacks samples")
    print("• Respects train/val/test splits from pre-computed CSV files")
    print("• Can apply filters (year, month, polarization, stripmap mode, etc.)")
    print("=" * 70)

if __name__ == "__main__":
    simple_test()
