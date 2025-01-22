from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

def plot_results(original: np.ndarray, sinogram: np.ndarray, 
                fbp: np.ndarray, bp: np.ndarray,
                metrics_fbp: Tuple[float, float, float],
                metrics_bp: Tuple[float, float, float],
                title_suffix: str = "") -> None:
    """Enhanced visualization with custom titles"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    
    images = [
        (original, f"Original Phantom\n{title_suffix}"),
        (sinogram, "Sinogram (Radon Transform)"),
        (fbp, f"Filtered Back Projection\nMSE: {metrics_fbp[0]:.4f}, SSIM: {metrics_fbp[2]:.3f}"),
        (bp, f"Simple Back Projection\nMSE: {metrics_bp[0]:.4f}, SSIM: {metrics_bp[2]:.3f}")
    ]
    
    for ax, (img, title) in zip(axs.flat, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()