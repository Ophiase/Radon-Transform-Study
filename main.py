import os
import argparse
import numpy as np
from typing import Optional
from constants import SYNTHETIC_DIR, NUM_SAMPLES, IMAGE_SIZE, NOISE_LEVEL, THETA
from synthetic_data import generate_dataset
from dicom_io import save_sinogram_dicom, load_dicom
from radon_transform import compute_sinogram, filtered_back_projection, simple_back_projection
from metrics import calculate_metrics
from visualization import plot_results


def process_sample(phantom: np.ndarray, sample_id: int, sample_type: str) -> None:
    """Full processing pipeline for a single CT sample"""
    print(f"\nProcessing {sample_type} sample {sample_id}")

    # Compute Radon transform
    sinogram = compute_sinogram(phantom, THETA)

    # Save sinogram
    sinogram_path = os.path.join(
        SYNTHETIC_DIR, f"sinogram_{sample_id}_{sample_type}.dcm")
    save_sinogram_dicom(sinogram, sinogram_path)

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
        title_suffix=f"Sample {sample_id} ({sample_type})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CT Scan Processing Pipeline")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic dataset")
    parser.add_argument("--process", type=int,
                        help="Process specific sample by ID (0-based)")
    args = parser.parse_args()

    if args.generate:
        print("Generating synthetic CT data...")
        generate_dataset(SYNTHETIC_DIR, NUM_SAMPLES, IMAGE_SIZE, NOISE_LEVEL)
        print(f"Generated {NUM_SAMPLES} synthetic CT pairs in {SYNTHETIC_DIR}")
        return

    if args.process is not None:
        if args.process >= NUM_SAMPLES:
            print(f"Error: Sample index must be between 0 and {NUM_SAMPLES-1}")
            return

        TO_PROCESS = [
            # "clean",
            "noisy"
        ]

        for sample_type in TO_PROCESS:
            file_path = os.path.join(
                SYNTHETIC_DIR,
                f"phantom_{args.process}_{sample_type}.dcm"
            )

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            phantom = load_dicom(file_path)
            process_sample(phantom, args.process, sample_type)

        return

    print("Please specify either --generate or --process <sample_id>")


if __name__ == "__main__":
    main()
