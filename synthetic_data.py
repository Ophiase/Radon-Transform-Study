import os
import numpy as np
from typing import Optional
from dicom_io import save_phantom_dicom

def generate_phantom(size: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a parameterized CT phantom with random shapes"""
    rng = np.random.default_rng(seed)
    phantom = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    # Random tissue structures
    for _ in range(rng.integers(2, 5)):
        shape_type = rng.choice(['sphere', 'cube', 'cylinder'])
        intensity = rng.uniform(0.1, 1.0)
        pos = (rng.integers(0, size), rng.integers(0, size))
        size_shape = rng.integers(10, size//4)
        
        y, x = np.ogrid[:size, :size]
        if shape_type == 'sphere':
            mask = (x - pos[0])**2 + (y - pos[1])**2 <= size_shape**2
        elif shape_type == 'cube':
            mask = (
                (x > pos[0] - size_shape//2) & (x < pos[0] + size_shape//2) &
                (y > pos[1] - size_shape//2) & (y < pos[1] + size_shape//2)
            )
        elif shape_type == 'cylinder':
            mask = (
                ((x - pos[0])**2 <= size_shape**2) &
                (np.abs(y - pos[1]) <= size_shape)
            )
        phantom[mask] = intensity
        
    return phantom

def add_gaussian_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Add realistic CT noise to the phantom"""
    noise = np.random.normal(0, noise_level * image.max(), image.shape)
    return np.clip(image + noise, 0, 1)

def generate_dataset(output_dir: str, num_samples: int, size: int, noise_level: float) -> None:
    """Generate and save a collection of synthetic CT scans"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in range(num_samples):
        # Generate clean phantom
        clean = generate_phantom(size, seed=idx)
        
        # Save clean version
        clean_path = os.path.join(output_dir, f'phantom_{idx}_clean.dcm')
        save_phantom_dicom(clean, clean_path)
        
        # Generate and save noisy version
        noisy = add_gaussian_noise(clean, noise_level)
        noisy_path = os.path.join(output_dir, f'phantom_{idx}_noisy.dcm')
        save_phantom_dicom(noisy, noisy_path)