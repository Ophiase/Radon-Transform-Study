from typing import Tuple
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio


def center_crop(original: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop original image to match target shape while preserving center"""
    if original.ndim != 2 or len(target_shape) != 2:
        raise ValueError("Inputs must be 2D images")

    # Calculate crop boundaries for each dimension
    crops = []
    for o_dim, t_dim in zip(original.shape, target_shape):
        if o_dim < t_dim:
            raise ValueError(
                f"Original dimension {o_dim} smaller than target {t_dim}")

        diff = o_dim - t_dim
        start = diff // 2
        end = start + t_dim
        crops.append((start, end))

    # Perform center cropping
    return original[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1]]


def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple:
    """Compute image similarity metrics"""

    if original.shape != reconstructed.shape:
        print("Warning: [calculate metrics] The shapes do not match.")
        cropped_original = center_crop(original, reconstructed.shape)
    else:
        cropped_original = original

    # Calculate metrics on aligned images
    mse = mean_squared_error(cropped_original, reconstructed)
    psnr = peak_signal_noise_ratio(cropped_original, reconstructed)
    ssim = structural_similarity(cropped_original, reconstructed,
                                 data_range=reconstructed.max()-reconstructed.min())

    return mse, psnr, ssim
