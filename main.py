import os
import argparse
import numpy as np
from typing import Final, List, Optional
from constants import (
    SYNTHETIC_DIR, REAL_DATA_DIR, NUM_SAMPLES,
    IMAGE_SIZE, NOISE_LEVEL, THETA
)
from synthetic_data import generate_dataset
from data_downloader import download_real_ct_data
from dicom_io import save_sinogram_dicom, load_dicom
from radon_transform import compute_sinogram, filtered_back_projection, simple_back_projection
from metrics import calculate_metrics
from visualization import plot_results

###################################################################################

# Constants for processing configuration
SYNTHETIC_PROCESS_MODES: Final[List[str]] = [
    # "clean",
    "noisy"
]
REAL_PROCESS_MODES: Final[List[str]] = ["original"]

###################################################################################


def validate_sample_id(sample_id: str, data_type: str) -> bool:
    """Validate sample ID based on data type"""
    if data_type == "real":
        if "_" not in sample_id:
            print(f"Invalid real data format. Use 'X_XXX' format (e.g., 1_008)")
            return False
        return True
    else:
        try:
            id_num = int(sample_id)
            if 0 <= id_num < NUM_SAMPLES:
                return True
            print(
                f"Synthetic sample index must be between 0 and {NUM_SAMPLES-1}")
            return False
        except ValueError:
            print(f"Invalid synthetic sample ID: {sample_id}")
            return False


def process_phantom(phantom: np.ndarray, data_path: str,
                    sample_id: str, process_mode: str) -> None:
    """Core processing pipeline for a single phantom"""
    try:
        print(f"\nProcessing sample {sample_id} ({process_mode})")

        # Compute Radon transform
        sinogram = compute_sinogram(phantom, THETA)

        # Save sinogram with mode differentiation
        sinogram_dir = os.path.join(data_path, "sinograms")
        os.makedirs(sinogram_dir, exist_ok=True)
        sinogram_file = f"sinogram_{sample_id.replace('_', '-')}_{process_mode}.dcm"
        save_sinogram_dicom(sinogram, os.path.join(
            sinogram_dir, sinogram_file))

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


def process_real_data_sample(data_path: str, sample_id: str) -> None:
    """Handle processing of real CT cases"""
    try:
        case_num, file_num = sample_id[0:1], sample_id[1:1]
        case_dir = f"case{case_num}"
        file_prefix = f"case{case_num}{file_num}"

        case_path = os.path.join(data_path, case_dir)
        if not os.path.isdir(case_path):
            raise FileNotFoundError(f"Case directory not found: {case_path}")

        # Find matching DICOM file
        matches = [f for f in os.listdir(case_path)
                   if f.startswith(file_prefix) and f.endswith('.dcm')]

        if not matches:
            raise FileNotFoundError(f"No DICOM files found for {sample_id}")

        file_path = os.path.join(case_path, matches[0])
        phantom = load_dicom(file_path)
        process_phantom(phantom, data_path, sample_id, "original")

    except Exception as e:
        print(f"Error processing real data sample: {str(e)}")
        raise


def process_synthetic_data_sample(data_path: str, sample_id: int) -> None:
    """Handle all processing modes for synthetic data"""
    for mode in SYNTHETIC_PROCESS_MODES:
        try:
            file_name = f"phantom_{sample_id}_{mode}.dcm"
            file_path = os.path.join(data_path, file_name)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"DICOM file not found: {file_path}")

            phantom = load_dicom(file_path)
            process_phantom(phantom, data_path, str(sample_id), mode)

        except Exception as e:
            print(f"Error processing {mode} mode: {str(e)}")
            continue

###################################################################################


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
    parser.add_argument("--process", type=str,
                        help="Process sample by ID (format: N for synthetic, X_XXX for real)")
    parser.add_argument("--data-type", choices=["synthetic", "real"], default="synthetic",
                        help="Type of data to process")

    try:
        args = parser.parse_args()
        data_path = REAL_DATA_DIR if args.data_type == "real" else SYNTHETIC_DIR

        if args.generate:
            print("Generating synthetic CT data...")
            generate_dataset(SYNTHETIC_DIR, NUM_SAMPLES,
                             IMAGE_SIZE, NOISE_LEVEL)
            print(
                f"Generated {NUM_SAMPLES} synthetic CT pairs in {SYNTHETIC_DIR}")
            return

        if args.download:
            print("Downloading real CT data...")
            download_real_ct_data()
            return

        if args.process is not None:
            if not os.path.isdir(data_path):
                raise NotADirectoryError(
                    f"Data directory not found: {data_path}")

            if not validate_sample_id(args.process, args.data_type):
                return

            if args.data_type == "real":
                process_real_data_sample(data_path, args.process)
            else:
                process_synthetic_data_sample(data_path, int(args.process))
            return

        parser.print_help()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
