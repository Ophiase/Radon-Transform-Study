# radon_transform.py
import numpy as np
from typing import Optional
from skimage.transform import radon as sk_radon, iradon as sk_iradon
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import rotate

###################################################################################

DEFAULT_USE_LIBRARY_RADON: bool = True # Works
DEFAULT_USE_LIBRARY_FBP: bool = True # TODO (not working)
DEFAULT_USE_LIBRARY_BP: bool = True # ??? (both don't work great)

###################################################################################


def compute_sinogram(
    image: np.ndarray,
    theta: np.ndarray,
    use_library: bool = DEFAULT_USE_LIBRARY_RADON
) -> np.ndarray:
    """Compute Radon transform with implementation choice"""
    if use_library:
        return sk_radon(image, theta=theta, circle=False)
    return _radon_custom(image, theta)


def filtered_back_projection(
    sinogram: np.ndarray,
    theta: np.ndarray,
    size: int,
    use_library: bool = DEFAULT_USE_LIBRARY_FBP
) -> np.ndarray:
    """Filtered back projection with implementation choice"""
    if use_library:
        return sk_iradon(sinogram, theta=theta, filter_name='ramp', output_size=size, circle=False)

    # Custom implementation
    scaled_sino = np.clip(sinogram, -1e6, 1e6)  # Prevent overflow
    return _back_project(scaled_sino, theta, filtered=True)[:size, :size]


def simple_back_projection(
    sinogram: np.ndarray,
    theta: np.ndarray,
    size: int,
    use_library: bool = DEFAULT_USE_LIBRARY_BP
) -> np.ndarray:
    """Unfiltered back projection with implementation choice"""
    if use_library:
        return sk_iradon(sinogram, theta=theta, filter_name=None, output_size=size, circle=False)

    # Custom implementation
    scaled_sino = np.clip(sinogram, -1e6, 1e6)
    return _back_project(scaled_sino, theta, filtered=False)[:size, :size]


###################################################################################


def _radon_custom(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Custom Radon transform implementation"""
    sinogram = np.zeros((image.shape[1], len(theta)))

    for i, angle in enumerate(theta):
        rotated = rotate(image, -angle, reshape=False, order=1)
        sinogram[:, i] = rotated.sum(axis=0)

    return sinogram


def _ramp_filter(projection_size: int) -> np.ndarray:
    """Create frequency-domain ramp filter"""
    freq = fftfreq(projection_size).reshape(-1, 1)
    return np.abs(freq) * 2  # Ram-Lak filter


def _back_project(sinogram: np.ndarray, theta: np.ndarray, filtered: bool) -> np.ndarray:
    """Core back projection algorithm"""
    reconstruction = np.zeros((sinogram.shape[0], sinogram.shape[0]))
    x_center = sinogram.shape[0] // 2

    # integral over (proj,angle) in R^2
    for i, (proj, angle) in enumerate(zip(sinogram.T, theta)):
        if filtered:
            # Frequency domain filtering
            f_proj = fft(proj)
            filtered_proj = np.real(ifft(f_proj * _ramp_filter(len(proj))))
        else:
            filtered_proj = proj

        # Create 2D projection
        backproj = np.tile(filtered_proj, (sinogram.shape[0], 1))
        rotated = rotate(backproj, angle, reshape=False, order=1)
        reconstruction += rotated

    return reconstruction * np.pi / (2 * len(theta))
