"""
Test script to verify the dataloader online mode fixes.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from maya4 import get_sar_dataloader

# Define data directory
DATA_DIR = os.path.abspath('data')
print(f'Data directory: {DATA_DIR}')
print(f'Directory exists: {os.path.exists(DATA_DIR)}')

# Check existing files
if os.path.exists(DATA_DIR):
    zarr_files = list(Path(DATA_DIR).rglob('*.zarr'))
    print(f'Found {len(zarr_files)} Zarr files locally')
    if zarr_files:
        print('Sample files:')
        for zf in zarr_files[:3]:
            print(f'  - {zf}')
else:
    print('Creating data directory...')
    os.makedirs(DATA_DIR, exist_ok=True)

print('\n' + '='*80)
print('Testing dataloader with online=True and verbose=True')
print('='*80 + '\n')

try:
    # Create a SAR dataloader with online mode enabled
    loader = get_sar_dataloader(
        data_dir=DATA_DIR,
        level_from='rcmc',
        level_to='az',
        batch_size=16,
        num_workers=0,
        patch_mode='rectangular',
        patch_size=(1, 1000),
        buffer=(1000, 1000),
        stride=(1, 1000),
        shuffle_files=False,
        patch_order='chunk',
        complex_valued=True,
        save_samples=False,
        backend='zarr',
        verbose=True,  # Enable verbose to see what's happening
        samples_per_prod=100,  # Small number for testing
        cache_size=100,
        online=True,  # Will attempt to download from HuggingFace
        max_products=1,  # Just test with 1 product
        use_balanced_sampling=False  # Disable for simple test - needs minimum ~10 products
    )
    
    print('\n' + '='*80)
    print('Dataloader created successfully!')
    print('='*80)
    print(f'\nDataset found {len(loader.dataset._files)} files')
    
    if len(loader.dataset._files) > 0:
        print('\nFirst file info:')
        print(loader.dataset._files.iloc[0])
        
        # Try to iterate through one batch
        print('\n' + '='*80)
        print('Testing batch iteration...')
        print('='*80 + '\n')
        
        for i, (x_batch, y_batch) in enumerate(loader):
            print(f'✓ Batch {i}: x_batch shape {x_batch.shape}, y_batch shape {y_batch.shape}')
            if i >= 2:  # Just test first 3 batches
                break
        
        print('\n' + '='*80)
        print('✓ SUCCESS: All tests passed!')
        print('='*80)
    else:
        print('\n⚠ Warning: No files found in dataset')

except Exception as e:
    print(f'\n✗ ERROR: {type(e).__name__}: {e}')
    import traceback
    print('\nFull traceback:')
    traceback.print_exc()
    print('\n' + '='*80)
    print('TEST FAILED')
    print('='*80)
    sys.exit(1)
