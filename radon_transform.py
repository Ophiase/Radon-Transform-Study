import numpy as np
from skimage.transform import radon, iradon

def compute_sinogram(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute Radon transform using skimage's implementation"""
    return radon(image, theta=theta, circle=False)

def filtered_back_projection(sinogram: np.ndarray, theta: np.ndarray, size: int) -> np.ndarray:
    """Standard filtered back projection"""
    return iradon(sinogram, theta=theta, filter_name='ramp', output_size=size, circle=False)

def simple_back_projection(sinogram: np.ndarray, theta: np.ndarray, size: int) -> np.ndarray:
    """Unfiltered back projection for comparison"""
    return iradon(sinogram, theta=theta, filter_name=None, output_size=size, circle=False)