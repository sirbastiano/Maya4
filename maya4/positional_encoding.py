import torch 
from typing import Tuple, Optional
import numpy as np

class PositionalEncoding: 
    def __init__(self, complex_valued: bool = False, concat: bool = True):
        self.complex_valued = complex_valued
        self.concat = concat
    def forward(self, x: torch.Tensor, position: Tuple[int, int], d_model: int) -> torch.Tensor:
        pass
    def inverse(self, x: torch.Tensor, position: Tuple[int, int], d_model: int) -> torch.Tensor:
        pass
class PositionalEncoding2D(PositionalEncoding):
    def __init__(self, complex_valued: bool = False, concat: bool = True):
        super().__init__(complex_valued, concat)
        self._y_positions = None
        self._x_positions = None
        self._pos_cache_key = None

    def forward(self, inp: np.ndarray, position: Tuple[int, int], max_length: Tuple[int, int]) -> np.ndarray:
        """
        Add 2D positional encoding to the input numpy array.

        Args:
            inp (np.ndarray): Input array of shape (ph, pw, ch) or (ph, pw).
            position (Tuple[int, int]): Tuple of (y_offset, x_offset) for global positioning.
            max_length (Tuple[int, int]): Maximum length tuple (max_y, max_x) for normalization.

        Returns:
            np.ndarray: Output array with position embeddings appended.
        """
        y_offset, x_offset = position
        max_y, max_x = max_length
        ph, pw = inp.shape[:2]
        
        if inp.ndim == 2:
            inp = inp.reshape(ph, pw, 1)
        elif inp.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input patch for positional encoding. Got patch of shape: {inp.shape}")
        
        ph, pw, ch = inp.shape
        
        # Check if cached position arrays exist and have correct dimensions
        cache_key = (ph, pw)
        
        if (self._y_positions is None or 
            self._x_positions is None or 
            self._pos_cache_key != cache_key):
            
            # Recompute position arrays
            self._y_positions = np.repeat(np.arange(ph), pw).reshape(ph, pw, 1)
            self._x_positions = np.tile(np.arange(pw), ph).reshape(ph, pw, 1)
            self._pos_cache_key = cache_key
        
        # Add global offsets to cached arrays
        global_y_positions = y_offset + self._y_positions
        global_x_positions = x_offset + self._x_positions
        
        # Normalize positions to [0, 1] range
        y_position_embedding = (global_y_positions / max_y)
        x_position_embedding = (global_x_positions / max_x)
        
        if np.iscomplexobj(inp):
            # Create a complex positional embedding: real=x, imag=y
            pos_embedding = x_position_embedding[..., 0] + 1j * y_position_embedding[..., 0]
            pos_embedding = pos_embedding[..., np.newaxis]
            out = np.concatenate((inp, pos_embedding), axis=-1)
        else:
            out = np.concatenate((inp, y_position_embedding, x_position_embedding), axis=-1)
        
        return out
    def inverse(self, x: torch.Tensor, position: Tuple[int, int], d_model: int) -> torch.Tensor:
        """
        Remove 2D positional encoding from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width, d_model).
            position (Tuple[int, int]): Tuple of (y, x) positions.
            d_model (int): Dimension of the model.
        Returns:
            torch.Tensor: Tensor with positional encoding removed.
        """
        pass

class PositionalEncodingRow(PositionalEncoding):
    def __init__(self, complex_valued: bool = False, concat: bool = True):
        super().__init__(complex_valued, concat)
        self._y_positions = None
        self._pos_cache_key = None

    def forward(self, inp: np.ndarray, position: Tuple[int, int], max_length: Tuple[int, int]) -> np.ndarray:
        """
        Add row-based positional encoding to the input numpy array (only y-position).

        Args:
            inp (np.ndarray): Input array of shape (ph, pw, ch) or (ph, pw).
            position (Tuple[int, int]): Tuple of (y_offset, x_offset) for global positioning.
            max_length (Tuple[int, int]): Maximum length tuple (max_y, max_x) for normalization.

        Returns:
            np.ndarray: Output array with row position embeddings appended.
        """
        y_offset, x_offset = position
        max_y, max_x = max_length
        ph, pw = inp.shape[:2]
        
        if inp.ndim == 2:
            inp = inp.reshape(ph, pw, 1)
        elif inp.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input patch for positional encoding. Got patch of shape: {inp.shape}")
        
        ph, pw, ch = inp.shape
        
        # Check if cached position arrays exist and have correct dimensions
        cache_key = (ph, pw)
        
        if (self._y_positions is None or 
            self._pos_cache_key != cache_key):
            
            # Recompute position arrays (only y)
            self._y_positions = np.repeat(np.arange(ph), pw).reshape(ph, pw, 1)
            self._pos_cache_key = cache_key
        
        # Add global offsets to cached arrays
        global_y_positions = y_offset + self._y_positions
        
        # Normalize positions to [0, 1] range
        y_position_embedding = (global_y_positions / max_y)
        
        if np.iscomplexobj(inp):
            # Create a complex positional embedding with only imaginary part
            pos_embedding = 1j * y_position_embedding[..., 0]
            pos_embedding = pos_embedding[..., np.newaxis]
            out = np.concatenate((inp, pos_embedding), axis=-1)
        else:
            out = np.concatenate((inp, y_position_embedding), axis=-1)
        
        return out
    def inverse(self, x: torch.Tensor, position: Tuple[int, int], d_model: int) -> torch.Tensor:
        """
        Remove 2D positional encoding from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height, width, d_model).
            position (Tuple[int, int]): Tuple of (y, x) positions.
            d_model (int): Dimension of the model.
        Returns:
            torch.Tensor: Tensor with positional encoding removed.
        """
        pass

def create_positional_encoding_module(name: str, complex_valued: bool = False, concat: bool = True) -> Optional[PositionalEncoding]:
    if name == "2d":
        return PositionalEncoding2D(complex_valued=complex_valued, concat=concat)
    elif name == "row":
        return PositionalEncodingRow(complex_valued=complex_valued, concat=concat)
    elif name == "none" or name is None or name is False:
        return None
    else:
        raise ValueError(f"Unknown positional encoding name: {name}")