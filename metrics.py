from typing import Tuple
import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error, peak_signal_noise_ratio

def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float, float]:
    """Compute image similarity metrics"""
    mse = mean_squared_error(original, reconstructed)
    psnr = peak_signal_noise_ratio(original, reconstructed)
    ssim = structural_similarity(original, reconstructed, data_range=1.0)
    return mse, psnr, ssim