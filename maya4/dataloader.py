import functools
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from maya4.api import download_metadata_from_product, list_base_files_in_repo, list_repos_by_author
from maya4.caching import ChunkCache
from maya4.coords_generation import LazyCoordinateGenerator, LazyCoordinateRange
from maya4.normalization import SARTransform
from maya4.positional_encoding import PositionalEncoding2D, PositionalEncodingAzimuth, PositionalEncodingRange, create_positional_encoding_module
from maya4.utils import (
    GT_MAX,
    GT_MIN,
    RC_MAX,
    RC_MIN,
    SampleFilter,
    extract_stripmap_mode_from_filename,
    get_balanced_sample_files,
    get_part_from_filename,
    get_sample_visualization,
    parse_product_filename,
)

# Export classes for external use
__all__ = [
    "SARZarrDataset",
    "KPatchSampler",
    "SARDataloader",
    "get_sar_dataloader",
    "LazyCoordinateRange",
    "LazyCoordinateGenerator",
    "ChunkCache",
    "PositionalEncoding2D",
    "PositionalEncodingAzimuth",
    "PositionalEncodingRange",
    "create_positional_encoding_module",
]


class SARZarrDataset(Dataset):
    """
    PyTorch Dataset for loading SAR (Synthetic Aperture Radar) data patches from Zarr format archives.

    This class supports efficient patch sampling from multiple Zarr files, with both rectangular and parabolic patch extraction. It handles both local and remote (Hugging Face) Zarr stores, with on-demand patch downloading and LRU chunk caching for performance.

    Features:
        - Loads SAR data patches from Zarr stores (local or remote), supporting real and complex-valued data.
        - Multiple patch sampling modes: "rectangular", "parabolic".
        - Efficient patch coordinate indexing and caching for fast repeated access.
        - Optional patch transformation and visualization utilities.
        - Handles both input and target SAR processing levels (e.g., "rcmc" and "az").
        - Supports saving/loading patch indices to avoid recomputation.
        - Implements chunk-level LRU caching for efficient repeated access.
        - Flexible filtering by part, year, month, polarization, and stripmap mode via SampleFilter.
        - Supports positional encoding and concatenation of patches.

    Args:
        data_dir (str): Directory containing Zarr files.
        filters (SampleFilter, optional): Filter for selecting data by part, year, etc.
        author (str, optional): Author or dataset identifier. Defaults to 'Maya4'.
        online (bool, optional): If True, enables remote Hugging Face access. Defaults to False.
        return_whole_image (bool, optional): If True, returns the whole image as a single patch. Defaults to False.
        transform (callable, optional): Optional transform to apply to both input and target patches.
        patch_size (Tuple[int, int], optional): Size of the patch (height, width). Defaults to (256, 256). If the patch width or height is set to a negative value, it will be computed based on the image dimensions minus the buffer.
        complex_valued (bool, optional): If True, returns complex-valued tensors. If False, returns real and imaginary parts stacked. Defaults to False.
        level_from (str, optional): Key for the input SAR processing level. Defaults to "rcmc".
        level_to (str, optional): Key for the target SAR processing level. Defaults to "az".
        patch_mode (str, optional): Patch extraction mode: "rectangular", or "parabolic". Defaults to "rectangular".
        save_samples (bool, optional): If True, saves computed patch indices to disk. Defaults to True.
        buffer (Tuple[int, int], optional): Buffer (margin) to avoid sampling near image edges. Defaults to (100, 100).
        stride (Tuple[int, int], optional): Stride for patch extraction. Defaults to (50, 50).
        max_base_sample_size (Tuple[int, int], optional): Maximum base sample size. Defaults to (-1, -1).
        verbose (bool, optional): If True, prints verbose output. Defaults to True.
        cache_size (int, optional): Maximum number of chunks to cache in memory.
        positional_encoding (bool, optional): If True, adds positional encoding to input patches. Defaults to True.
        dataset_length (int, optional): Optional override for dataset length.
        max_products (int, optional): Maximum number of Zarr products to use. Defaults to 10.
        samples_per_prod (int, optional): Number of patches to sample per product. Defaults to 1000.
        concatenate_patches (bool, optional): If True, concatenates patches along the specified axis.
        concat_axis (int, optional): Axis along which to concatenate patches.
        max_stripmap_modes (int, optional): Maximum number of stripmap modes.
        use_positional_as_token (bool, optional): If True, uses positional encoding as a token.

    Attributes:
        data_dir (str): Directory containing Zarr files.
        patch_size (Tuple[int, int]): Patch size (height, width).
        level_from (str): Input SAR processing level.
        level_to (str): Target SAR processing level.
        patch_mode (str): Patch extraction mode.
        buffer (Tuple[int, int]): Buffer for patch extraction.
        stride (Tuple[int, int]): Stride for patch extraction.
        cache_size (int): LRU cache size for chunk loading.
        positional_encoding (bool): Whether to add positional encoding.
        dataset_length (int): Optional override for dataset length.
        ... (see code for additional attributes)

    Example:
        >>> dataset = SARZarrDataset("/path/to/zarrs", patch_size=(128, 128), cache_size=1000)
        >>> x, y = dataset[("path/to/zarr", 100, 100)]
        >>> dataset.visualize_item(("path/to/zarr", 100, 100))
    """

    def __init__(
        self,
        data_dir: str,
        filters: Optional[SampleFilter] = None,
        author: str = "Maya4",
        online: bool = False,
        transform: Optional[SARTransform] = None,
        patch_size: Tuple[int, int] = (256, 256),
        complex_valued: bool = False,
        level_from: str = "rcmc",
        level_to: str = "az",
        save_samples: bool = True,
        buffer: Tuple[int, int] = (100, 100),
        stride: Tuple[int, int] = (50, 50),
        block_pattern: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
        cache_size: int = 1000,
        positional_encoding: str = "2d",
        max_products: int = 10,
        samples_per_prod: int = 1000,
        max_stripmap_modes: int = 6,
        use_balanced_sampling: bool = True,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.filters = filters if filters is not None else SampleFilter()
        self.author = author
        self.transform = transform
        self._patch_size = patch_size
        self.level_from = level_from
        self.level_to = level_to

        self.complex_valued = complex_valued
        assert buffer[0] >= 0 and buffer[1] >= 0, "Buffer values must be non-negative"
        self.buffer = buffer
        assert stride[0] >= -1 and stride[1] >= -1, "Stride values must be positive"
        self.stride = stride
        self.block_pattern = block_pattern
        self.verbose = verbose
        self.save_samples = save_samples
        self.online = online
        self.positional_encoding = positional_encoding
        self._max_products = max_products
        self._samples_per_prod = samples_per_prod
        self.max_stripmap_modes = max_stripmap_modes
        self.use_balanced_sampling = use_balanced_sampling 
        assert not (self.use_balanced_sampling is True and self.online is False), "Balanced sampling requires online=True to access remote data"
        self.split = split

        self._patch: Dict[str, np.ndarray] = {self.level_from: np.array([0]), self.level_to: np.array([0])}
        self.positional_encoding_module = create_positional_encoding_module(
            positional_encoding, complex_valued=complex_valued, concat=True
        )

        # Initialize chunk cache component
        self.chunk_cache = ChunkCache(
            self,
            cache_size=cache_size,
            online=online,
            data_dir=str(self.data_dir),
            author=self.author,
            verbose=self.verbose,
        )

        self._y_coords: Dict[os.PathLike, np.ndarray] = {}
        self._x_coords: Dict[os.PathLike, np.ndarray] = {}
        self._stores: Dict[os.PathLike, Dict[str, zarr.Array]] = {}
        self._samples_by_file: Dict[os.PathLike, List[Tuple[int, int]]] = {}

        # DataFrame to store file metadata and lazy coordinates
        self._files: Optional[pd.DataFrame] = None

        self._initialize_stores()
        if self.verbose:
            print(
                f"Initialized dataloader with config: buffer={buffer}, stride={stride}, patch_size={patch_size}, complex_values={complex_valued}"
            )

    def _get_base_sample(self, zfile: os.PathLike, y: int, x: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get base sample patches using the chunk cache.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            y (int): y-coordinate.
            x (int): x-coordinate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target patches.
        """
        ph, pw = self.get_patch_size(zfile)

        # Use chunk cache to get samples
        patch_from = self.chunk_cache.get_sample(zfile, self.level_from, y, x, ph, pw)
        patch_to = self.chunk_cache.get_sample(zfile, self.level_to, y, x, ph, pw)

        return patch_from, patch_to

    def get_patch_size(self, zfile: Optional[Union[str, os.PathLike]]) -> Tuple[int, int]:
        """
        Retrieve the patch size for a given Zarr file.
        Resolves -1 values to actual dimensions from the zarr store.

        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            Tuple[int, int]: Patch size (height, width) with -1 resolved to actual dimensions.
        """
        ph, pw = self._patch_size

        # If patch size contains -1, resolve it to actual dimensions
        if ph == -1 or pw == -1:
            if zfile is None:
                raise ValueError("Cannot resolve patch size with -1 without a zfile reference")

            arr = self.get_store_at_level(Path(zfile), self.level_from)
            height, width = arr.shape[:2]

            if ph == -1:
                ph = height
            if pw == -1:
                pw = width

        return (ph, pw)

    def get_whole_sample_shape(self, zfile: os.PathLike) -> Tuple[int, int]:
        """
        Get the full dimensions of a zarr sample.

        Args:
            zfile (os.PathLike): Path to the Zarr file.

        Returns:
            Tuple[int, int]: Height and width of the sample.
        """
        arr = self.get_store_at_level(Path(zfile), self.level_from)
        return arr.shape[:2]

    def get_store_at_level(self, zfile: Union[os.PathLike, str], level: str) -> zarr.Array:
        """
        Get the zarr array at a specific processing level.
        Opens the store lazily if not already opened.
        Downloads metadata from HuggingFace if online and not available locally.

        Args:
            zfile (Union[os.PathLike, str]): Path to the Zarr file.
            level (str): Processing level name.

        Returns:
            zarr.Array: The zarr array at the specified level.
        """
        zfile = Path(zfile)
        level_dir = zfile / level

        # Check if we need to download metadata from HuggingFace
        if not zfile.exists() or not level_dir.exists():
            if self.online:
                zfile_name = os.path.basename(zfile)
                part = get_part_from_filename(zfile)
                repo_id = f"{self.author}/{part}"
                if self.verbose:
                    print(f"Downloading metadata for {zfile_name} from {repo_id}...")
                download_metadata_from_product(
                    zfile_name=str(zfile_name),
                    local_dir=os.path.join(self.data_dir, part),
                    levels=[level, self.level_from, self.level_to],
                    repo_id=repo_id,
                )

        # Check if store is already opened in DataFrame
        if self._files is not None:
            idx = self._files.index[self._files["full_name"] == zfile]
            if len(idx) > 0:
                store = self._files.at[idx[0], "store"]
                # Lazy load: open store if not already opened
                if store is None:
                    if zfile.exists() or self.online:
                        try:
                            store = zarr.open(str(zfile), mode="r")
                            self._files.at[idx[0], "store"] = store
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Could not open {zfile}: {e}")
                            # Fallback to direct open
                            return zarr.open(str(level_dir), mode="r")

                # Return the level array from the opened store
                if store is not None:
                    return store[level]

        # Fallback: direct open (for files not in DataFrame)
        return zarr.open(str(level_dir), mode="r")

    def _build_file_list(self):
        """
        Build the list of zarr files to use, either from local directory or remote HuggingFace.
        Creates a pandas DataFrame with file metadata.
        """
        if self.online:
            # Get files from remote HuggingFace repositories
            repos = list_repos_by_author(self.author)
            if self.verbose:
                print(f"Found {len(repos)} repositories by author '{self.author}': {repos}")

            remote_files = {}
            for repo in repos:
                repo_files = list_base_files_in_repo(repo_id=repo)
                if self.verbose:
                    print(f"Found {len(repo_files)} files in remote repository: '{repo}'")
                repo_name = repo.split("/")[-1]
                remote_files[repo_name] = repo_files

            # Parse filenames to get metadata
            records = [
                r
                for r in (
                    parse_product_filename(os.path.join(self.data_dir, part, f))
                    for part, files in remote_files.items()
                    for f in files
                )
                if r is not None
            ]
            if self.verbose:
                print(f"Total files found in remote repository: {len(records)}")
            df = pd.DataFrame(records)
        else:
            # Get files from local directory
            all_files = []
            if self.data_dir.exists():
                for part_dir in self.data_dir.iterdir():
                    if part_dir.is_dir():
                        for zfile in part_dir.glob("*.zarr"):
                            all_files.append(zfile)

            if self.verbose:
                print(f"Found {len(all_files)} local zarr files")

            # Parse filenames to get metadata
            records = [r for r in (parse_product_filename(str(f)) for f in all_files) if r is not None]
            df = pd.DataFrame(records)

        # Handle empty DataFrame
        if len(df) == 0:
            self._files = pd.DataFrame(columns=["full_name", "store", "samples"])
            if self.verbose:
                print("No files found after parsing")
            return

        # Filter files using SampleFilter
        self._files = self.filters.filter_products(df).copy()
        self._files.sort_values(by=["full_name"], inplace=True)

        # Apply balanced sampling if enabled
        if self.use_balanced_sampling:
            balanced_files = get_balanced_sample_files(
                max_samples=self._max_products,
                data_dir=self.data_dir,
                sample_filter=self.filters,
                config_path=str(self.data_dir),
                verbose=self.verbose,
                split_type=self.split,
                repos=self.filters.parts if self.filters.parts else ["PT1", "PT2", "PT3", "PT4"],
            )
            if balanced_files:
                balanced_paths = [Path(f) for f in balanced_files]
                self._files = self._files[self._files["full_name"].isin(balanced_paths)]
                if self.verbose:
                    print(
                        f"Applied balanced sampling: selected {len(self._files)} files from {len(balanced_files)} balanced candidates"
                    )
            else:
                if self.verbose:
                    print("Warning: Balanced sampling returned no files, falling back to standard selection")
                self._files = self._files.iloc[: self._max_products]
        else:
            self._files = self._files.iloc[: self._max_products]

        # Add columns for stores and samples
        self._files["store"] = None
        self._files["samples"] = None

        if self.verbose:
            print(f"Selected {len(self._files)} files after filtering")
            print(f"Files: {self._files['full_name'].tolist()}")

    def _initialize_stores(self):
        """
        Initialize file list without opening stores (lazy loading).
        Stores are opened on-demand when first accessed.
        """
        t0 = time.time()
        self._build_file_list()
        if self.verbose:
            dt = time.time() - t0
            print(f"File list calculation took {dt:.2f} seconds.")

        # Don't open stores yet - do it lazily when needed
        # Just initialize the DataFrame columns
        if self.verbose:
            print(f"Deferred zarr store opening for {len(self._files)} files (lazy loading)")

    def get_files(self) -> List[os.PathLike]:
        """
        Get list of all zarr files in dataset.

        Returns:
            List[os.PathLike]: List of file paths.
        """
        if self._files is None or len(self._files) == 0:
            return []
        return self._files["full_name"].tolist()

    def get_samples_by_file(
        self, zfile: Union[str, os.PathLike]
    ) -> Union[List[Tuple[int, int]], LazyCoordinateGenerator]:
        """
        Get patch coordinates for a specific file.

        Args:
            zfile (Union[str, os.PathLike]): Path to the Zarr file.

        Returns:
            Union[List[Tuple[int, int]], LazyCoordinateGenerator]: Lazy coordinate generator or list of coordinates.
        """
        if self._files is None:
            return []

        try:
            samples = self._files.loc[self._files["full_name"] == Path(zfile), "samples"].values
            if len(samples) > 0:
                return samples[0] if samples[0] is not None else []
            return []
        except Exception:
            return []

    def calculate_patches_from_store(
        self,
        zfile: os.PathLike,
        patch_order: str = "row",
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        """
        Calculate patch coordinates for a zarr store using lazy generation.

        Args:
            zfile (os.PathLike): Path to the Zarr file.
            patch_order (str): Order of patch extraction ("row", "col", or "chunk").
            window (Optional): Optional window for patch extraction ((y_start, y_end), (x_start, x_end)).
        """
        zfile = Path(zfile)

        # Get the shape of the zarr array
        try:
            arr = self.get_store_at_level(zfile, self.level_from)
            height, width = arr.shape[:2]
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get shape for {zfile}: {e}")
            return

        # Get patch size
        ph, pw = self.get_patch_size(zfile)

        # Calculate valid coordinate ranges with buffer and stride
        if window is not None:
            (y_start, y_end), (x_start, x_end) = window
        else:
            y_start = self.buffer[0]
            y_end = height - self.buffer[0] - ph
            x_start = self.buffer[1]
            x_end = width - self.buffer[1] - pw

        # Create lazy coordinate ranges
        y_range = LazyCoordinateRange(y_start, y_end, self.stride[0])
        x_range = LazyCoordinateRange(x_start, x_end, self.stride[1])

        # Create lazy coordinate generator
        lazy_coords = LazyCoordinateGenerator(
            y_range=y_range,
            x_range=x_range,
            patch_order=patch_order,
            block_pattern=self.block_pattern,
            zfile=zfile,
            dataset=self,
        )

        # Store in DataFrame
        idx = self._files.index[self._files["full_name"] == zfile]
        if len(idx) > 0:
            self._files.at[idx[0], "samples"] = lazy_coords
            if self.verbose:
                print(f"Generated {len(lazy_coords)} lazy coordinates for {zfile.name}")

    def visualize_item(
        self,
        idx: Tuple[str, int, int],
        show: bool = True,
        vminmax: Optional[Union[Tuple[float, float], str]] = (0, 1000),
        figsize: Tuple[int, int] = (15, 15),
    ) -> None:
        """
        Visualizes a data sample at the specified index using matplotlib.
        This method retrieves the input and target data corresponding to the given index,
        generates their magnitude visualizations, and displays them side by side as images.
        The visualization can be shown interactively or closed after creation.
        Args:
            idx (Tuple[str, int, int]): The index tuple specifying the data sample to visualize.
            show (bool, optional): If True, displays the plot interactively. If False, closes the figure after creation. Defaults to True.
            vminmax (Optional[Union[Tuple[float, float], str]], optional): Minimum and maximum values for color scaling, or a string (e.g., 'raw') for automatic scaling. Defaults to (0, 1000).
            figsize (Tuple[int, int], optional): Size of the matplotlib figure. Defaults to (15, 15).
        Returns:
            None
        """
        import matplotlib.pyplot as plt

        zfile, y, x = idx
        x, y = self._get_base_sample(Path(zfile), y, x)  # x, y)
        imgs = []
        img, vmin, vmax = get_sample_visualization(data=x, plot_type="magnitude", vminmax=vminmax)
        imgs.append({"name": self.level_from, "img": img, "vmin": vmin, "vmax": vmax})
        img, vmin, vmax = get_sample_visualization(data=y, plot_type="magnitude", vminmax=vminmax)
        imgs.append({"name": self.level_to, "img": img, "vmin": vmin, "vmax": vmax})

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for i in range(2):
            im = axes[i].imshow(
                imgs[i]["img"], aspect="auto", cmap="viridis", vmin=imgs[i]["vmin"], vmax=imgs[i]["vmax"]
            )

            axes[i].set_title(f"{imgs[i]['name'].upper()} product")
            axes[i].set_xlabel("Range")
            axes[i].set_ylabel("Azimuth")
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            # axis equal aspect ratio
            axes[i].set_aspect("equal", adjustable="box")

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)

    def __len__(self) -> int:
        """
        Get the total number of patches across all files.

        Returns:
            int: Total number of patches.
        """
        if self._files is None:
            return 0

        total = 0
        for idx, row in self._files.iterrows():
            samples = row["samples"]
            if samples is not None:
                total += len(samples)
        return total

    def __getitem__(self, idx: Tuple[str, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a patch from the dataset given a (zfile, y, x) tuple.

        Args:
            idx (Tuple[str, int, int]): A tuple containing the zfile path name, y-coordinate, and x-coordinate
                specifying the location of the patch to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target patches as torch tensors.
                The format and type of the tensors depend on the `complex_valued` attribute:
                    - If `complex_valued` is False, tensors have shape (2, patch_height, patch_width) with real and imaginary parts stacked.
                    - If `complex_valued` is True, tensors retain their complex dtype.

        Raises:
            IndexError: If the provided index is out of bounds.
            KeyError: If the specified levels are not found in the zarr store.

        Notes:
            - If `patch_mode` is "parabolic", patches are sampled using the `_sample_parabolic_patch` method.
            - If a `transform` is provided, it is applied to both input and target patches.
        """
        zfile, y, x = idx
        # Extract stride number from filename if present
        stripmap_mode = extract_stripmap_mode_from_filename(os.path.basename(zfile))
        zfile = Path(zfile)
        start_time = time.time()

        t0 = time.time()

        # Retrieve the horizontal size (width) from the store for self.level_from
        sample_height, sample_width = self.get_whole_sample_shape(zfile)
        if self.verbose:
            dt = time.time() - t0
            print(
                f"Sample shape for {zfile} at level {self.level_from}: {sample_height}x{sample_width} took {dt:.4f} seconds"
            )

        t0 = time.time()
        patch_from, patch_to = self._get_base_sample(zfile, y, x)
        if self.verbose:
            dt = time.time() - t0
            print(f"Base sample loading for {zfile} at ({y}, {x}) took {dt:.4f} seconds")

        if self.transform:
            t0 = time.time()
            patch_from = self.transform(patch_from, self.level_from)
            patch_to = self.transform(patch_to, self.level_to)
            if self.verbose:
                dt = time.time() - t0
                print(f"Patch transformation took {dt:.4f} seconds")
        # print(f"Patch shape before stacking: {patch_from.shape}, {patch_to.shape}")
        if not self.complex_valued:
            t0 = time.time()
            patch_from = np.stack((np.real(patch_from), np.imag(patch_from)), axis=-1).astype(np.float32)
            patch_to = np.stack((np.real(patch_to), np.imag(patch_to)), axis=-1).astype(np.float32)
            if self.verbose:
                dt = time.time() - t0
                print(f"Complex to real conversion took {dt:.4f} seconds")
        # print(f"Shape before positional encoding: {patch_from.shape}")
        if self.positional_encoding:
            t0 = time.time()
            global_column_index = x + stripmap_mode * sample_width

            patch_from = self.positional_encoding_module.forward(
                patch_from, (y, global_column_index), (sample_height, sample_width * self.max_stripmap_modes)
            )
            patch_to = self.positional_encoding_module.forward(
                patch_to, (y, global_column_index), (sample_height, sample_width * self.max_stripmap_modes)
            )
            if self.verbose:
                dt = time.time() - t0
                print(f"Patch positional encoding took {dt:.4f} seconds")

        # Convert to tensors
        x_tensor = torch.from_numpy(patch_from)
        y_tensor = torch.from_numpy(patch_to)

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Loading patch ({zfile}, {y}, {x}) took {elapsed:.4f} seconds. Stripmap mode: {stripmap_mode}")

        return x_tensor, y_tensor


class KPatchSampler(Sampler):
    """
    PyTorch Sampler that yields (file_idx, y, x) tuples for patch sampling.
    Draws k patches from each file in round-robin, with optional shuffling of files and patches.

    Args:
        dataset (SARZarrDataset): The dataset to sample patches from.
        samples_per_prod (int): Number of patches to sample per file. If 0, all patches are sampled.
        shuffle_files (bool): Whether to shuffle the order of files between epochs.
        shuffle_patches (bool): Whether to shuffle the patches within each file.
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, prints timing and sampling info.
        max_products (int): Maximum number of products to sample from.
        patch_order (str): Order in which to sample patches from each file. Options are "row", "col", or "chunk".
        samples_per_prod (int): Number of patches to sample per product. If 0, all patches are sampled.
    """

    def __init__(
        self,
        dataset: SARZarrDataset,
        samples_per_prod: int = 0,
        shuffle_files: bool = True,
        seed: int = 42,
        verbose: bool = True,
        patch_order: str = "row",
    ):
        self.dataset = dataset
        self.samples_per_prod = samples_per_prod
        self.shuffle_files = shuffle_files
        self.patch_order = patch_order
        self.seed = seed
        self.verbose = verbose
        self.beginning = True
        self.coords: Dict[Path, List[Tuple[int, int]]] = {}
        self.zfiles = None

    def __iter__(self):
        """
        Iterate over the dataset, yielding (file_idx, y, x) tuples for patch sampling.
        Shuffles files and/or patches if enabled.
        """
        self.beginning = False
        rng = np.random.default_rng(self.seed)
        files = self.dataset.get_files()  # files.copy()
        if self.verbose:
            print(files)
        if self.shuffle_files:
            rng.shuffle(files)
        for idx, f in enumerate(files):
            # Mark patches as loaded for this file
            if self.zfiles is not None and idx not in self.zfiles and f not in self.zfiles:
                if self.verbose:
                    print(f"Skipping file {f} as it's not in the filtered list.")
                continue
            if self.dataset.get_samples_by_file(f) is not None and len(self.dataset.get_samples_by_file(f)) == 0:
                # print(f"Calculating patches for file {f}")
                self.dataset.calculate_patches_from_store(f, patch_order=self.patch_order)
            self.coords[Path(f)] = self.get_coords_from_store(f)
            t0 = time.time()
            for y, x in self.coords[Path(f)]:
                if self.verbose:
                    print(f"Sampling from file {f}, patch at ({y}, {x})")
                yield (f, y, x)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"Sampling {len(self.coords[Path(f)])} patches from file {f} took {elapsed:.2f} seconds.")

    def get_coords_from_store(
        self, zfile: Union[str, os.PathLike], window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ):
        if self.dataset.get_samples_by_file(zfile) is not None and len(self.dataset.get_samples_by_file(zfile)) == 0:
            # print(f"Calculating patches for file {zfile}")
            self.dataset.calculate_patches_from_store(Path(zfile), patch_order=self.patch_order, window=window)
        lazy_coords = self.dataset.get_samples_by_file(zfile)

        # If samples_per_prod is specified, limit the coordinates
        if self.samples_per_prod > 0:
            limited_coords = []
            for i, coord in enumerate(lazy_coords):
                if i >= self.samples_per_prod:
                    break
                limited_coords.append(coord)
            return limited_coords
        else:
            # Return the full lazy generator
            return lazy_coords

    def filter_by_zfiles(self, zfiles: Union[List[Union[str, os.PathLike]], str, int]) -> None:
        """
        Filter the dataset to only include samples from the specified list of zarr files.

        Args:
            zfiles (List[Union[str, os.PathLike]]): List of zarr file paths to include.
        """
        if isinstance(zfiles, (str, int, Path)):
            zfiles = [zfiles]
        self.zfiles = [Path(zf) if not isinstance(zf, (Path, int)) else zf for zf in zfiles]

    def __len__(self):
        """Return the total number of samples to be drawn by the sampler."""
        if self.beginning:
            return len(self.dataset)
        else:
            total = 0
            for zfile in self.dataset.get_files():
                lazy_coords = self.dataset.get_samples_by_file(zfile)
                if self.samples_per_prod > 0:
                    total += min(self.samples_per_prod, len(lazy_coords))
                else:
                    total += len(lazy_coords)
            return total


class SARDataloader(DataLoader):
    dataset: SARZarrDataset

    def __init__(
        self,
        dataset: SARZarrDataset,
        batch_size: int,
        sampler: KPatchSampler,
        num_workers: int = 2,
        pin_memory: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory
        )
        self.verbose = verbose

    def get_coords_from_zfile(
        self, zfile: Union[str, os.PathLike], window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        return self.sampler.get_coords_from_store(zfile, window=window)

    def filter_by_zfiles(self, zfiles: Union[List[Union[str, os.PathLike]], str, int]) -> None:
        """
        Filter the dataset to only include samples from the specified list of zarr files.

        Args:
            zfiles (List[Union[str, os.PathLike]]): List of zarr file paths to include.
        """
        self.sampler.filter_by_zfiles(zfiles)
        if self.verbose:
            print(f"Filtered dataset to {len(self.dataset)} samples from {len(zfiles)} files.")


def get_sar_dataloader(
    data_dir: str,
    filters: Optional[SampleFilter] = None,
    batch_size: int = 8,
    num_workers: int = 2,
    transform: Optional[Callable] = None,
    level_from: str = "rcmc",
    level_to: str = "az",
    patch_size: Tuple[int, int] = (512, 512),
    buffer: Tuple[int, int] = (100, 100),
    stride: Tuple[int, int] = (50, 50),
    block_pattern: Optional[Tuple[int, int]] = None,
    positional_encoding: str = "2d",
    shuffle_files: bool = True,
    patch_order: str = "row",  # "row", "col", or "chunk"
    complex_valued: bool = False,
    save_samples: bool = True,
    verbose: bool = True,
    cache_size: int = 10000,
    max_products: int = 10,
    samples_per_prod: int = 0,
    online: bool = True,
    use_balanced_sampling: bool = True,
    split: str = "train",
) -> SARDataloader:
    """
    Create and return a PyTorch DataLoader for SAR data using SARZarrDataset and KPatchSampler.

    Args:
        data_dir (str): Path to the directory containing SAR data.
        file_pattern (str, optional): Glob pattern for Zarr files. Defaults to "*.zarr".
        batch_size (int, optional): Number of samples per batch. Defaults to 8.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 2.
        return_whole_image (bool, optional): If True, returns the whole image. Defaults to False.
        transform (callable, optional): Optional transform to apply to samples.
        level_from (str, optional): Input SAR processing level. Defaults to "rcmc".
        level_to (str, optional): Target SAR processing level. Defaults to "az".
        patch_mode (str, optional): Patch extraction mode. Defaults to "rectangular".
        patch_size (Tuple[int, int], optional): Patch size. Defaults to (512, 512).
        buffer (Tuple[int, int], optional): Buffer size. Defaults to (100, 100).
        stride (Tuple[int, int], optional): Stride for patch extraction. Defaults to (50, 50).
        positional_encoding (bool, optional): If True, adds positional encoding to patches. Defaults to True.
        shuffle_files (bool, optional): Shuffle file order. Defaults to True.
        patch_order (str, optional): Patch extraction order. Defaults to "row".
        complex_valued (bool, optional): If True, loads data as complex-valued. Defaults to False.
        save_samples (bool, optional): If True, saves sampled patches. Defaults to True.
        cache_size (int, optional): LRU cache size for chunks. Defaults to 10000.
        max_products (int, optional): Maximum number of products. Defaults to 10.
        samples_per_prod (int, optional): Number of patches per product. Defaults to 0 (all patches).
        online (bool, optional): If True, uses online data loading. Defaults to True.
        verbose (bool, optional): If True, prints additional info. Defaults to True.
        geographic_clustering (bool, optional): If True, clusters data by geographic location. Defaults to False.
        n_clusters (int, optional): Number of geographic clusters when clustering is enabled. Defaults to 10.
        split (str, optional): Dataset split to use (e.g., "train", "val", "test"). Defaults to "train".

    Returns:
        SARDataloader: PyTorch DataLoader for the SAR dataset.
    """
    dataset = SARZarrDataset(
        data_dir=data_dir,
        filters=filters,
        transform=transform,
        patch_size=patch_size,
        block_pattern=block_pattern,
        complex_valued=complex_valued,
        level_from=level_from,
        level_to=level_to,
        save_samples=save_samples,
        buffer=buffer,
        stride=stride,
        verbose=verbose,
        cache_size=cache_size,
        online=online,
        max_products=max_products,
        samples_per_prod=samples_per_prod,
        positional_encoding=positional_encoding,
        use_balanced_sampling=use_balanced_sampling,
        split=split,
    )
    sampler = KPatchSampler(
        dataset,
        samples_per_prod=samples_per_prod,
        shuffle_files=shuffle_files,
        patch_order=patch_order,
        verbose=verbose,
    )
    return SARDataloader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False, verbose=verbose
    )


# Example usage
if __name__ == "__main__":

    # Create SARTransform using the factory method
    transforms = SARTransform.create_minmax_normalized_transform(
        normalize=True, rc_min=RC_MIN, rc_max=RC_MAX, gt_min=GT_MIN, gt_max=GT_MAX, complex_valued=True
    )

    loader = get_sar_dataloader(
        data_dir="/Data/sar_focusing",
        level_from="rc",
        level_to="az",
        batch_size=4,
        num_workers=4,
        patch_mode="rectangular",
        complex_valued=False,
        shuffle_files=False,
        patch_order="col",
        transform=transforms,
        max_base_sample_size=(-1, -1),

    )
    for i, (x_batch, y_batch) in enumerate(loader):
        print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
        # break
