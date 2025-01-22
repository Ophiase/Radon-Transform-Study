# 🧬 Simple Radon Transform Study

This project applies the Radon Transform in medical imaging, focusing on synthetic CT scan generation, sinogram computation, and image reconstruction. It includes filtered and simple back projection methods and evaluates reconstruction quality using various metrics.

## Features

- **Synthetic Data Generation**: Create CT phantoms with random shapes and noise.
- **Radon Transform**: Compute sinograms.
    - Optional custom implementation
- **Image Reconstruction**: Perform filtered and simple back projections.
    - Optional custom implementation
- **Metrics Calculation**: Evaluate reconstruction quality using MSE, PSNR, and SSIM.
- **Visualization**: Visualize images with Plotly and Matplotlib.

## Usage

### Get Data

```bash
python3 main.py --generate # to generate
python3 main.py --download # to download real data
```

### Process a Specific Sample

To process a sample by its ID:

```bash
# Process synthetic sample 2 (both clean/noisy)
python main.py --process 2 --data-type synthetic

# Process real sample 0
python main.py --process 0 --data-type real # not working yet
```

- Result on ``--process 2 --data-type synthetic``
    - <img src="./resources/experiment.png" width=300>

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- scikit-image
- Plotly
- Matplotlib
- pydicom

## DICOM data

- https://www.visus.com/en/downloads/jivex-dicom-viewer.html
    - Implemented
- https://singularhealth-my.sharepoint.com/personal/jhill_singular_health/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjhill%5Fsingular%5Fhealth%2FDocuments%2F3Dicom%20%2D%20DICOM%20Library&ga=1
    - TODO