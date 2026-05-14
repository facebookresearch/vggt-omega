import os
import unittest
from pathlib import Path
import tempfile

import cv2
import numpy as np

import Imath
import OpenEXR
import io

## Tell OpenCV to enable reading and writing EXR files
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _store_exr(
    fname: str | Path,
    data: np.ndarray,
    params: list | None = None,
) -> bool:
    """
    Writes a NumPy array as an EXR file using OpenCV.
    Assumes input data is already np.float32.
    """
    if Path(fname).suffix != ".exr":
        raise ValueError(
            f"Only filenames with suffix .exr allowed but received: {fname}"
        )
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError(
            f"Image needs to contain two or three channels but received: {data.shape}"
        )
    return cv2.imwrite(str(fname), data, params if params else [])



def _store_depth(fname: str | Path, data: np.ndarray) -> bool:
    """
    Stores a depth map from a NumPy array into an EXR file.

    Args:
        fname (str or Path): The filename to save the depth map to.
        data (numpy.ndarray): The depth map to save.

    Returns:
        bool: True if the depth map was saved successfully, False otherwise.
    """
    # Ensure data is float32, as required by OpenCV for EXR saving
    data_np = data.astype(np.float32)

    data_np = data_np.squeeze()  # remove all 1-dim entries
    if data_np.ndim != 2:
        raise ValueError(f"Depth map needs to be 2D, but received: {data_np.shape}")

    # use 32-bit float with PIZ compression for depth maps
    params = [
        cv2.IMWRITE_EXR_TYPE,
        cv2.IMWRITE_EXR_TYPE_FLOAT,
        cv2.IMWRITE_EXR_COMPRESSION,
        cv2.IMWRITE_EXR_COMPRESSION_PIZ,
    ]
    return _store_exr(fname, data_np, params=params)



def _load_exr(fname: str | Path) -> np.ndarray:
    """Reads an EXR image file into a NumPy array using OpenCV."""
    if Path(fname).suffix != ".exr":
        raise ValueError(
            f"Only filenames with suffix .exr allowed but received: {fname}"
        )
    data = cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Failed to read EXR file: {fname}")
    return data

def _load_depth(fname: str | Path) -> np.ndarray:
    """
    Loads a depth map from an EXR file into a NumPy array.

    Args:
        fname (str or Path): The filename of the EXR file to load.

    Returns:
        The loaded depth map as a NumPy array.
    """
    data = _load_exr(fname)
    if data.ndim != 2:
        raise ValueError(f"Depth map needs to be 2D, but loaded: {data.shape}")
    return data




# def exr_to_npy(exr_path, channel="Z", fs_wrapper=None):
#     # Open file
#     if fs_wrapper is not None:
#         with fs_wrapper.open(str(exr_path), "rb") as f:
#             exr_file = OpenEXR.InputFile(io.BytesIO(f.read()))
#     else:
#         exr_file = OpenEXR.InputFile(str(exr_path))

#     # Get resolution from header
#     dw = exr_file.header()["dataWindow"]
#     w = dw.max.x - dw.min.x + 1
#     h = dw.max.y - dw.min.y + 1

#     # Read the Z channel as float32
#     pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
#     raw_data = exr_file.channel(channel, pixel_type)

#     # Convert bytes to numpy array
#     arr = np.frombuffer(raw_data, dtype=np.float32)
#     arr = arr.reshape((h, w))

#     return arr
