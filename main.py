import numpy as np
from constants import IMAGE_SIZE, THETA
from synthetic_data import generate_phantom
from radon_transform import compute_sinogram, filtered_back_projection, simple_back_projection
from metrics import calculate_metrics
from visualization import plot_results
from dicom_io import save_sinogram_dicom

def main() -> None:
    print("Generate synthetic CT data")
    phantom = generate_phantom(IMAGE_SIZE)
    
    print("Compute Radon transform")
    sinogram = compute_sinogram(phantom, THETA)
    
    print("Save synthetic DICOM (demonstration only)")
    save_sinogram_dicom(sinogram, "synthetic_sinogram.dcm")
    
    print("Reconstruction")
    fbp_recon = filtered_back_projection(sinogram, THETA, IMAGE_SIZE)
    bp_recon = simple_back_projection(sinogram, THETA, IMAGE_SIZE)
    
    print("Calculate metrics")
    metrics_fbp = calculate_metrics(phantom, fbp_recon)
    metrics_bp = calculate_metrics(phantom, bp_recon)
    
    print("Visualize results")
    plot_results(phantom, sinogram, fbp_recon, bp_recon, metrics_fbp, metrics_bp)

if __name__ == "__main__":
    main()