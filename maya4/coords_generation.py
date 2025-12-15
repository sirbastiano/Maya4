import os
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from maya4.dataloader import SARZarrDataset

class LazyCoordinateRange:
    """
    Lazy replacement for np.arange that doesn't materialize the array.
    """
    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self._length = max(0, (stop - start + step - 1) // step)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")
        return self.start + index * self.step
    
    def __iter__(self):
        current = self.start
        while current < self.stop:
            yield current
            current += self.step

class LazyCoordinateGenerator:
    """
    Lazy generator for coordinate tuples that generates coordinates on-demand.
    Supports different ordering patterns (row, col, chunk) and block patterns.
    """
    
    def __init__(self, y_range: LazyCoordinateRange, x_range: LazyCoordinateRange, 
                 patch_order: str = "row", block_pattern: Optional[Tuple[int, int]] = None,
                 zfile: Optional[os.PathLike] = None, dataset: Optional['SARZarrDataset'] = None):
        self.y_range = y_range
        self.x_range = x_range
        self.patch_order = patch_order
        assert patch_order in ["row", "col", "chunk"], f"Invalid patch_order: {patch_order}"
        self.block_pattern = block_pattern
        self.zfile = zfile
        self.dataset = dataset
        
        # Pre-calculate total length without materializing arrays
        self._length = len(y_range) * len(x_range)
    
    def __iter__(self):
        """Generate coordinates lazily based on the specified order and block pattern."""
        if self.patch_order == "row":
            yield from self._generate_row_order()
        elif self.patch_order == "col":
            yield from self._generate_col_order()
        elif self.patch_order == "chunk":
            yield from self._generate_chunk_order()
        else:
            raise ValueError(f"Unknown patch_order: {self.patch_order}")
    
    def _generate_row_order(self):
        """Generate coordinates in row-major order with optional block pattern."""
        if self.block_pattern is not None:
            yield from self._generate_block_pattern_row()
        else:
            for y in self.y_range:
                for x in self.x_range:
                    yield (y, x)
    
    def _generate_col_order(self):
        """Generate coordinates in column-major order with optional block pattern."""
        if self.block_pattern is not None:
            yield from self._generate_block_pattern_col()
        else:
            for x in self.x_range:
                for y in self.y_range:
                    yield (y, x)
    
    def _generate_chunk_order(self):
        """Generate coordinates in chunk-aware order for optimal cache performance."""
        if self.dataset is None or self.zfile is None:
            yield from self._generate_row_order()
            return
        
        try:
            # Get chunk information
            sample = self.dataset.get_store_at_level(self.zfile, self.dataset.level_from)
            ch, cw = sample.chunks
            
            # Group coordinates by chunks without materializing all coordinates
            chunk_coords = {}
            for y in self.y_range:
                for x in self.x_range:
                    cy, cx = y // ch, x // cw
                    chunk_key = (cy, cx)
                    if chunk_key not in chunk_coords:
                        chunk_coords[chunk_key] = []
                    chunk_coords[chunk_key].append((y, x))
            
            # Sort chunk keys and yield coordinates chunk by chunk
            sorted_chunks = sorted(chunk_coords.keys())
            for chunk_key in sorted_chunks:
                coords = chunk_coords[chunk_key]
                # Sort coordinates within chunk (x first, then y for cache locality)
                coords.sort(key=lambda coord: (coord[1], coord[0]))
                for coord in coords:
                    yield coord
                    
        except Exception:
            yield from self._generate_row_order()
    
    def _generate_block_pattern_row(self):
        """Generate coordinates with block pattern in row-major order."""
        if self.block_pattern is None:
            raise ValueError("block_pattern must be set for block pattern generation.")
        block_size, _ = self.block_pattern
        
        for y in self.y_range:
            # Process x coordinates in blocks
            x_start = 0
            while x_start < len(self.x_range):
                x_end = min(x_start + block_size, len(self.x_range))
                for x_idx in range(x_start, x_end):
                    x = self.x_range[x_idx]  # Get actual x coordinate
                    yield (y, x)
                x_start = x_end
        
    def _generate_block_pattern_col(self):
        """Generate coordinates with block pattern in column-major order."""
        if self.block_pattern is None:
            raise ValueError("block_pattern must be set for block pattern generation.")
        block_size, _ = self.block_pattern
        
        for x in self.x_range:
            # Process y coordinates in blocks
            y_start = 0
            while y_start < len(self.y_range):
                y_end = min(y_start + block_size, len(self.y_range))
                for y_idx in range(y_start, y_end):
                    y = self.y_range[y_idx]  # Get actual y coordinate
                    yield (y, x)
                y_start = y_end
    
    def __len__(self):
        """Return total number of coordinates."""
        return self._length
    
    def __getitem__(self, index: int) -> Tuple[int, int]:
        """
        Support indexing for compatibility with existing code.
        Note: This is less efficient than iteration for large datasets.
        """
        if index < 0:
            index += len(self)
        if index >= len(self) or index < 0:
            raise IndexError("Index out of range")
        
        # Calculate coordinates without materializing arrays
        if self.patch_order == "row":
            y_idx = index // len(self.x_range)
            x_idx = index % len(self.x_range)
            return (self.y_range[y_idx], self.x_range[x_idx])
        elif self.patch_order == "col":
            x_idx = index // len(self.y_range)
            y_idx = index % len(self.y_range)
            return (self.y_range[y_idx], self.x_range[x_idx])
        else:
            # For chunk order, fall back to list conversion (less efficient)
            coords = list(self)
            return coords[index]
