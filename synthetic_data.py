import numpy as np

def generate_phantom(size: int) -> np.ndarray:
    """Generate a custom CT phantom with geometric shapes"""
    phantom = np.zeros((size, size), dtype=np.float32)
    
    # Main circle (soft tissue)
    y, x = np.ogrid[:size, :size]
    center = size//2
    radius = size//4
    phantom[(x-center)**2 + (y-center)**2 <= radius**2] = 0.8
    
    # Bone-like rectangle
    phantom[center-30:center+30, center-10:center+10] = 1.0
    
    # Air pocket
    phantom[center-20:center+20, center-60:center-40] = 0.1
    return phantom