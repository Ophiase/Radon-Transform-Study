# 🧬 Simple Radon Transform Study

This project applies the Radon Transform in medical imaging, focusing on synthetic CT scan generation, sinogram computation, and image reconstruction. It includes filtered and simple back projection methods and evaluates reconstruction quality using various metrics.

## Features

- **Synthetic Data Generation**: Create CT phantoms with random shapes and noise.
- **Radon Transform**: Compute sinograms.
- **Image Reconstruction**: Perform filtered and simple back projections.
- **Metrics Calculation**: Evaluate reconstruction quality using MSE, PSNR, and SSIM.
- **Visualization**: Visualize images with Plotly and Matplotlib.

## Usage

### Generate Synthetic Data

To generate synthetic CT scans:

```bash
python main.py --generate
```

### Process a Specific Sample

To process a sample by its ID:

```bash
python main.py --process <sample_id>
```

Replace `<sample_id>` with the desired index.

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- scikit-image
- Plotly
- Matplotlib
- pydicom