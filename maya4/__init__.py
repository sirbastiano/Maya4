"""
Maya4 - SAR Data Processing and Dataloader Package

This package provides utilities for processing and loading Synthetic Aperture Radar (SAR)
data from Sentinel-1 missions. It includes efficient dataloaders, normalization modules,
and utilities for working with SAR data in Zarr format.

Main components:
- SARZarrDataset: PyTorch dataset for SAR data patches
- SARDataloader: Custom dataloader with efficient sampling
- SARTransform: Normalization and transformation modules
- API utilities for Hugging Face Hub integration
- Location utilities for spatial mapping
"""

__version__ = '0.1.0'
__author__ = 'Roberto Del Prete'

from .api import (
    download_file_from_hf,
    download_metadata_from_product,
    fetch_chunk_from_hf_zarr,
    filter_files_by_modalities,
    list_base_files_in_repo,
    list_files_in_repo,
    list_repos_by_author,
)

# Import main components using relative imports
from .dataloader import (
    KPatchSampler,
    LazyCoordinateGenerator,
    LazyCoordinateRange,
    SARDataloader,
    SARZarrDataset,
    get_sar_dataloader,
)
from .location_utils import (
    extract_location_from_zarr_filename_with_phidown,
    get_location_for_zarr_file,
    get_products_spatial_mapping,
    get_sar_product_locations,
)
from .normalization import (
    AdaptiveNormalizationModule,
    AdaptiveZScoreNormalize,
    BaseTransformModule,
    ComplexNormalizationModule,
    IdentityModule,
    NormalizationModule,
    RobustNormalize,
    SARTransform,
    ZScoreNormalize,
)
from .utils import (
    GT_MAX,
    GT_MIN,
    RC_MAX,
    RC_MIN,
    SampleFilter,
    create_dataloader_from_config,
    create_dataloaders,
    create_transforms_from_config,
    display_inference_results,
    extract_stripmap_mode_from_filename,
    get_balanced_sample_files,
    get_chunk_name_from_coords,
    get_part_from_filename,
    get_sample_visualization,
    get_zarr_version,
    minmax_inverse,
    minmax_normalize,
    parse_product_filename,
)

# dataset_creation module was removed/refactored
# from .dataset_creation import (
#     save_array_to_zarr,
#     dask_slice_saver,
# )

# Define public API
__all__ = [
    # Main classes
    'SARZarrDataset',
    'KPatchSampler',
    'SARDataloader',
    'LazyCoordinateRange',
    'LazyCoordinateGenerator',
    'get_sar_dataloader',
    
    # Normalization and transformation modules
    'SARTransform',
    'NormalizationModule',
    'ComplexNormalizationModule',
    'IdentityModule',
    'BaseTransformModule',
    'ZScoreNormalize',
    'AdaptiveZScoreNormalize',
    'RobustNormalize',
    'AdaptiveNormalizationModule',
    
    # API functions
    'list_base_files_in_repo',
    'list_repos_by_author',
    'fetch_chunk_from_hf_zarr',
    'download_metadata_from_product',
    'download_file_from_hf',
    'list_files_in_repo',
    'filter_files_by_modalities',
    
    # Utility functions
    'get_chunk_name_from_coords',
    'get_sample_visualization',
    'get_zarr_version',
    'minmax_normalize',
    'minmax_inverse',
    'extract_stripmap_mode_from_filename',
    'get_part_from_filename',
    'parse_product_filename',
    'SampleFilter',
    'display_inference_results',
    'get_balanced_sample_files',
    'create_transforms_from_config',
    'create_dataloader_from_config',
    'create_dataloaders',
    
    # Location utilities
    'get_products_spatial_mapping',
    'get_sar_product_locations',
    'extract_location_from_zarr_filename_with_phidown',
    'get_location_for_zarr_file',
    
    # Dataset creation - removed/refactored
    # 'save_array_to_zarr',
    # 'dask_slice_saver',
    
    # Constants
    'RC_MAX',
    'RC_MIN',
    'GT_MAX',
    'GT_MIN',
]

# Define what gets imported with "from dataloader import *"
__all__ = [
    # Main classes
    'SARZarrDataset',
    'KPatchSampler',
    'get_sar_dataloader',
    'SARDataloader',
    'SampleFilter',
    
    # Normalization and transformation modules
    'SARTransform', 
    'NormalizationModule',
    'ComplexNormalizationModule', 
    'IdentityModule',
    'BaseTransformModule',
    
    
    
    # API functions
    'list_base_files_in_repo',
    'fetch_chunk_from_hf_zarr', 
    'download_metadata_from_product',
    
    # Utility functions
    'get_products_spatial_mapping',
    'get_chunk_name_from_coords',
    'get_sample_visualization',
    'get_zarr_version',
    'minmax_normalize',
    'minmax_inverse',
    'extract_stripmap_mode_from_filename',
    
    # Constants
    'RC_MAX', 'RC_MIN', 'GT_MAX', 'GT_MIN'
]

# Version info
__version__ = "1.0.0"
__author__ = "Gabriele Daga"