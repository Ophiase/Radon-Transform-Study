import numpy as np
from skimage.transform import radon as sk_radon, iradon as sk_iradon
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import rotate

###################################################################################

DEFAULT_USE_LIBRARY_RADON: bool = True # Works
DEFAULT_USE_LIBRARY_FBP: bool = True # Works
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

    filtered_sino = _apply_ramp_filter(sinogram)
    return _back_project(filtered_sino, theta, size)


def simple_back_projection(
    sinogram: np.ndarray,
    theta: np.ndarray,
    size: int,
    use_library: bool = DEFAULT_USE_LIBRARY_BP
) -> np.ndarray:
    """Unfiltered back projection with implementation choice"""
    if use_library:
        return sk_iradon(sinogram, theta=theta, filter_name=None, output_size=size, circle=False)

    return _back_project(sinogram, theta, size)


###################################################################################


def _radon_custom(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Custom Radon transform implementation"""
    sinogram = np.zeros((image.shape[1], len(theta)))

    for i, angle in enumerate(theta):
        rotated = rotate(image, -angle, reshape=False, order=1)
        sinogram[:, i] = rotated.sum(axis=0)

    return sinogram


def _ramp_filter(projection_size: int) -> np.ndarray:
    """Create 1D frequency-domain ramp filter"""
    freq = fftfreq(projection_size)
    return np.abs(freq) * 2  # Correct 1D Ram-Lak filter


def _apply_ramp_filter(sinogram: np.ndarray) -> np.ndarray:
    """Apply ramp filter to each projection"""
    filtered_sino = np.zeros_like(sinogram)
    for i in range(sinogram.shape[1]):
        proj = sinogram[:, i]
        f_proj = fft(proj)
        filtered_proj = np.real(ifft(f_proj * _ramp_filter(len(proj))))
        filtered_sino[:, i] = filtered_proj
    return filtered_sino


def _back_project(sinogram: np.ndarray, theta: np.ndarray, size: int) -> np.ndarray:
    """Coordinate-based back projection avoiding rotation artifacts"""
    N = sinogram.shape[0]
    reconstruction = np.zeros((size, size))
    center = N // 2

    # Create grid centered at reconstruction center
    x = np.arange(size) - size//2
    y = np.arange(size) - size//2
    X, Y = np.meshgrid(x, y)

    for i, angle in enumerate(theta):
        proj = sinogram[:, i]
        theta_rad = np.deg2rad(angle)

        # Calculate detector positions for all points
        rot_X = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
        detector_pos = rot_X + center

        # Interpolate and accumulate
        interp_proj = np.interp(detector_pos.flatten(),
                                np.arange(N), proj, left=0, right=0)
        reconstruction += interp_proj.reshape(size, size)

    return reconstruction
