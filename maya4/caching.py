import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np

from maya4.api import fetch_chunk_from_hf_zarr
from maya4.utils import get_chunk_name_from_coords, get_part_from_filename

if TYPE_CHECKING:
    from maya4.dataloader import SARZarrDataset

class ChunkCache:
    """
    LRU cache component for loading and caching zarr chunks.
    Handles all chunk-level operations and caching logic.
    """
    
    def __init__(self, dataset: 'SARZarrDataset', cache_size: int = 1000, online: bool = True, data_dir: str = './data', author: str = 'default_author', verbose: bool = False):
        self.dataset = dataset
        self.cache_size = cache_size
        self.online = online
        self.data_dir = data_dir
        self.author = author
        self.verbose = verbose
        self._patch: Dict[str, np.ndarray] = {}
        
        # Setup LRU cache for chunk loading
        self._load_chunk = functools.lru_cache(maxsize=cache_size)(
            self._load_chunk_uncached
        )
    def _download_sample_if_missing(self, zfile: os.PathLike, level: str, y: int, x: int) -> Path:
        """
        Download a zarr chunk if it's missing locally (for online mode).
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.
            level (str): Processing level.
            y (int): y-coordinate.
            x (int): x-coordinate.
            
        Returns:
            Path: Path to the chunk file.
        """
        # Get chunk coordinates
        arr = self.dataset.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        cy, cx = y // ch, x // cw
        
        # Build chunk path
        part = get_part_from_filename(zfile)
        zfile_name = os.path.basename(zfile)
        chunk_name = get_chunk_name_from_coords(cy, cx, zfile, level)
        chunk_path = Path(self.data_dir) / part / zfile_name / level / chunk_name
        
        # Download if missing
        if not chunk_path.exists():
            repo_id = f"{self.author}/{part}"
            if self.verbose:
                print(f"Chunk {chunk_name} not found locally. Downloading from Hugging Face Zarr archive...")
            fetch_chunk_from_hf_zarr(level=level, y=y, x=x, zarr_archive=zfile_name, local_dir=os.path.join(self.data_dir, part), repo_id=repo_id)
        return chunk_path
    def _load_chunk_uncached(self, zfile: os.PathLike, level: str, cy: int, cx: int) -> np.ndarray:
        """
        Load a single chunk at chunk coordinates (cy, cx).
        This is the cacheable unit - individual chunks.
        
        Args:
            zfile (os.PathLike): Path to the Zarr file.
            level (str): Zarr group/level name.
            cy (int): Chunk y-index.
            cx (int): Chunk x-index.
            
        Returns:
            np.ndarray: Single chunk data.
        """
        arr = self.dataset.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Calculate actual array coordinates for this chunk
        chunk_y_start = cy * ch
        chunk_x_start = cx * cw
        chunk_y_end = min(chunk_y_start + ch, arr.shape[0])
        chunk_x_end = min(chunk_x_start + cw, arr.shape[1])
        
        # Ensure the chunk is downloaded if not already available
        if self.online:
            self._download_sample_if_missing(zfile, level, chunk_y_start, chunk_x_start)
        
        # Load the actual chunk data
        chunk_data = arr[chunk_y_start:chunk_y_end, chunk_x_start:chunk_x_end]
        
        return chunk_data.astype(np.complex128)
    
    def _get_sample_from_cached_chunks_vectorized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Optimized version that dispatches to specialized methods based on patch geometry.
        """
        arr = self.dataset.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Dispatch to optimized methods based on patch type
        if ph == 1:
            # Horizontal strip - most optimized
            return self._get_horizontal_strip_optimized(zfile, level, y, x, pw, ch, cw)
        elif pw == 1:
            # Vertical strip
            return self._get_vertical_strip_optimized(zfile, level, y, x, ph, ch, cw)
        elif ph <= ch and pw <= cw:
            # Small rectangular patch that likely fits in single chunk
            return self._get_small_patch_optimized(zfile, level, y, x, ph, pw, ch, cw)
        else:
            # Large rectangular patch - needs multi-chunk handling
            return self._get_large_patch_optimized(zfile, level, y, x, ph, pw, ch, cw)
    
    def _get_large_patch_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int, ch: int, cw: int) -> np.ndarray:
        """
        Optimized for large rectangular patches spanning multiple chunks.
        Uses direct extraction instead of creating large temporary arrays.
        """
        # Calculate chunk spans
        cy_start, cx_start = y // ch, x // cw
        cy_end, cx_end = (y + ph - 1) // ch, (x + pw - 1) // cw
        
        # Pre-allocate final patch only
        if level not in self._patch or self._patch[level].shape != (ph, pw):
            self._patch[level] = np.zeros((ph, pw), dtype=np.complex128)
        
        # Direct extraction without temporary arrays
        for cy in range(cy_start, cy_end + 1):
            for cx in range(cx_start, cx_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                
                # Calculate intersection bounds
                chunk_y_start, chunk_x_start = cy * ch, cx * cw
                chunk_y_end = chunk_y_start + chunk.shape[0]
                chunk_x_end = chunk_x_start + chunk.shape[1]
                
                # Global coordinates
                y_start = max(y, chunk_y_start)
                y_end = min(y + ph, chunk_y_end)
                x_start = max(x, chunk_x_start)
                x_end = min(x + pw, chunk_x_end)
                
                if y_start < y_end and x_start < x_end:
                    # Source indices in chunk
                    src_y1, src_y2 = y_start - chunk_y_start, y_end - chunk_y_start
                    src_x1, src_x2 = x_start - chunk_x_start, x_end - chunk_x_start
                    
                    # Destination indices in patch
                    dst_y1, dst_y2 = y_start - y, y_end - y
                    dst_x1, dst_x2 = x_start - x, x_end - x
                    
                    # Direct copy
                    self._patch[level][dst_y1:dst_y2, dst_x1:dst_x2] = chunk[src_y1:src_y2, src_x1:src_x2]

        return self._patch[level]
    
    def _get_small_patch_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int, ch: int, cw: int) -> np.ndarray:
        """
        Optimized for small patches that likely fit in a single chunk.
        """
        cy, cx = y // ch, x // cw
        chunk = self._load_chunk(zfile, level, cy, cx)
        
        dy, dx = y % ch, x % cw
        
        # Fast path: patch fits entirely in chunk
        if dy + ph <= chunk.shape[0] and dx + pw <= chunk.shape[1]:
            return chunk[dy:dy+ph, dx:dx+pw].copy()
        
        # Boundary case: use minimal allocation
        patch = np.zeros((ph, pw), dtype=np.complex128)
        max_h = min(ph, chunk.shape[0] - dy)
        max_w = min(pw, chunk.shape[1] - dx)
        
        if max_h > 0 and max_w > 0:
            patch[:max_h, :max_w] = chunk[dy:dy+max_h, dx:dx+max_w]
        
        return patch
    
    def _get_strip_optimized(self, zfile: os.PathLike, level: str, y: int, x: int, length: int, ch: int, cw: int, axis: int) -> np.ndarray:
        """
        Optimized extraction for strips (vertical or horizontal).
        axis=0: vertical strip (shape=(length, 1)), axis=1: horizontal strip (shape=(1, length))
        """
        if axis == 0:
            # Vertical strip: patch_size=(length, 1)
            cy_start, cy_end = y // ch, (y + length - 1) // ch
            cx = x // cw
            if level not in self._patch or self._patch[level].shape[0] != length:
                self._patch[level] = np.zeros(length, dtype=np.complex128)
            
            current_pos = 0
            for cy in range(cy_start, cy_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                chunk_y_start = cy * ch
                global_start = max(y, chunk_y_start)
                global_end = min(y + length, chunk_y_start + chunk.shape[0])
                if global_start < global_end:
                    dx = x % cw
                    if dx < chunk.shape[1]:
                        local_start = global_start - chunk_y_start
                        local_end = global_end - chunk_y_start
                        slice_height = local_end - local_start
                        self._patch[level][current_pos:current_pos + slice_height] = chunk[local_start:local_end, dx]
                        current_pos += slice_height
            return self._patch[level].reshape(length, 1)
        elif axis == 1:
            # Horizontal strip: patch_size=(1, length)
            cx_start, cx_end = x // cw, (x + length - 1) // cw
            cy = y // ch
            if level not in self._patch or self._patch[level].shape[0] != length:
                self._patch[level] = np.zeros(length, dtype=np.complex128)

            current_pos = 0
            for cx in range(cx_start, cx_end + 1):
                chunk = self._load_chunk(zfile, level, cy, cx)
                chunk_x_start = cx * cw
                global_start = max(x, chunk_x_start)
                global_end = min(x + length, chunk_x_start + chunk.shape[1])
                if global_start < global_end:
                    dy = y % ch
                    if dy < chunk.shape[0]:
                        local_start = global_start - chunk_x_start
                        local_end = global_end - chunk_x_start
                        slice_width = local_end - local_start
                        self._patch[level][current_pos:current_pos + slice_width] = chunk[dy, local_start:local_end]
                        current_pos += slice_width
            return self._patch[level].reshape(1, length)
        else:
            raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    def _get_vertical_strip_optimized(self, zfile, level, y, x, ph, ch, cw):
        return self._get_strip_optimized(zfile, level, y, x, ph, ch, cw, axis=0)

    def _get_horizontal_strip_optimized(self, zfile, level, y, x, pw, ch, cw):
        return self._get_strip_optimized(zfile, level, y, x, pw, ch, cw, axis=1)

    def _get_sample_from_cached_chunks(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Highly optimized version with fast single-chunk path.
        """
        arr = self.dataset.get_store_at_level(zfile, level)
        ch, cw = arr.chunks
        
        # Pre-calculate chunk coordinates
        cy_start, cx_start = y // ch, x // cw
        cy_end, cx_end = (y + ph - 1) // ch, (x + pw - 1) // cw
        
        # Fast path: single chunk (handles ~80-90% of cases)
        if cy_start == cy_end and cx_start == cx_end:
            chunk = self._load_chunk(zfile, level, cy_start, cx_start)
            dy, dx = y % ch, x % cw  
            
            # Bounds check only once
            if dy + ph <= chunk.shape[0] and dx + pw <= chunk.shape[1]:
                return chunk[dy:dy+ph, dx:dx+pw].copy()
            else:
                # Handle boundary case
                patch = np.zeros((ph, pw), dtype=np.complex128)
                max_h = min(ph, chunk.shape[0] - dy)
                max_w = min(pw, chunk.shape[1] - dx)
                patch[:max_h, :max_w] = chunk[dy:dy+max_h, dx:dx+max_w]
                return patch
        
        # Multi-chunk path (only when necessary)
        return self._get_sample_from_cached_chunks_vectorized(zfile, level, y, x, ph, pw)
    
    def get_sample(self, zfile: os.PathLike, level: str, y: int, x: int, ph: int, pw: int) -> np.ndarray:
        """
        Retrieve a sample patch from the Zarr store at the specified level and coordinates.
        This is the main entry point for getting patches with caching.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            level (str): Processing level (e.g., 'rcmc', 'az').
            y (int): y-coordinate of the patch.
            x (int): x-coordinate of the patch.
            ph (int): Patch height.
            pw (int): Patch width.

        Returns:
            np.ndarray: The desired patch as a NumPy array.
        """
        return self._get_sample_from_cached_chunks(zfile, level, y, x, ph, pw)

