import os
import argparse
import numpy as np
from typing import Final, List, Optional
from constants import (
    SYNTHETIC_DIR, REAL_DATA_DIR, NUM_SAMPLES,
    IMAGE_SIZE, NOISE_LEVEL, THETA, DOWNLOAD_URL
)
from synthetic_data import generate_dataset
from data_downloader import download_real_ct_data
from dicom_io import save_sinogram_dicom, load_dicom
from radon_transform import compute_sinogram, filtered_back_projection, simple_back_projection
from metrics import calculate_metrics
from visualization import plot_results

# Constants for processing configuration
SYNTHETIC_PROCESS_MODES: Final[List[str]] = [
    # "clean", 
    "noisy"
]
REAL_PROCESS_MODES: Final[List[str]] = ["original"]

def validate_sample_id(sample_id: int, max_samples: int) -> bool:
    """Ensure sample ID is within valid range with descriptive errors"""
    if not isinstance(sample_id, int):
        print(f"Error: Sample ID must be integer, got {type(sample_id)}")
        return False
    if 0 <= sample_id < max_samples:
        return True
    print(f"Error: Sample index must be between 0 and {max_samples-1}")
    return False

def process_phantom(phantom: np.ndarray, data_path: str, 
                   sample_id: int, process_mode: str) -> None:
    """Core processing pipeline for a single phantom"""
    try:
        print(f"\nProcessing sample {sample_id} ({process_mode})")
        
        # Compute Radon transform
        sinogram = compute_sinogram(phantom, THETA)
        
        # Save sinogram with mode differentiation
        sinogram_dir = os.path.join(data_path, "sinograms")
        os.makedirs(sinogram_dir, exist_ok=True)
        sinogram_file = f"sinogram_{sample_id}_{process_mode}.dcm"
        save_sinogram_dicom(sinogram, os.path.join(sinogram_dir, sinogram_file))
        
        # Reconstructions
        fbp_recon = filtered_back_projection(sinogram, THETA, IMAGE_SIZE)
        bp_recon = simple_back_projection(sinogram, THETA, IMAGE_SIZE)
        
        # Calculate metrics
        metrics_fbp = calculate_metrics(phantom, fbp_recon)
        metrics_bp = calculate_metrics(phantom, bp_recon)
        
        # Visualize results
        plot_results(
            phantom, sinogram, fbp_recon, bp_recon,
            metrics_fbp, metrics_bp,
            title_suffix=f"Sample {sample_id} ({process_mode})"
        )

    except Exception as e:
        print(f"Processing failed for {process_mode} sample: {str(e)}")
        raise

def process_data_sample(data_path: str, sample_id: int,
                       process_modes: List[str]) -> None:
    """Handle all processing modes for a single data sample"""
    for mode in process_modes:
        try:
            # Construct filename based on processing mode
            file_name = (f"phantom_{sample_id}_{mode}.dcm" if len(process_modes) > 1 
                        else f"phantom_{sample_id}.dcm")
            file_path = os.path.join(data_path, file_name)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"DICOM file not found: {file_path}")
            
            phantom = load_dicom(file_path)
            process_phantom(phantom, data_path, sample_id, mode)
            
        except Exception as e:
            print(f"Error processing {mode} mode: {str(e)}")
            continue

def main() -> None:
    """Professional-grade CLI with enhanced error resilience"""
    parser = argparse.ArgumentParser(
        description="Medical CT Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--generate", action="store_true",
                       help="Generate synthetic dataset")
    parser.add_argument("--download", action="store_true",
                       help="Download real CT dataset")
    parser.add_argument("--process", type=int,
                       help="Process sample by ID (0-based)")
    parser.add_argument("--data-type", choices=["synthetic", "real"], default="synthetic",
                       help="Type of data to process")

    try:
        args = parser.parse_args()
        data_path = SYNTHETIC_DIR if args.data_type == "synthetic" else REAL_DATA_DIR

        if args.generate:
            print("Generating synthetic CT data...")
            generate_dataset(SYNTHETIC_DIR, NUM_SAMPLES, IMAGE_SIZE, NOISE_LEVEL)
            print(f"Generated {NUM_SAMPLES} synthetic CT pairs in {SYNTHETIC_DIR}")
            return

        if args.download:
            print("Downloading real CT data...")
            download_real_ct_data(REAL_DATA_DIR, DOWNLOAD_URL)
            return

        if args.process is not None:
            # Validate data directory exists
            if not os.path.isdir(data_path):
                raise NotADirectoryError(f"Data directory not found: {data_path}")
            
            # Configure processing parameters
            process_modes = (SYNTHETIC_PROCESS_MODES if args.data_type == "synthetic" 
                            else REAL_PROCESS_MODES)
            max_samples = (NUM_SAMPLES if args.data_type == "synthetic" 
                          else len(os.listdir(REAL_DATA_DIR)))
            
            if validate_sample_id(args.process, max_samples):
                process_data_sample(data_path, args.process, process_modes)
            return

        parser.print_help()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()