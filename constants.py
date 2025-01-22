import os
from typing import Final
import numpy as np

# Path configuration
DATA_DIR: Final[str] = os.path.join(os.getcwd(), 'data')
SYNTHETIC_DIR: Final[str] = os.path.join(DATA_DIR, 'synthetic_data')

# Image parameters
IMAGE_SIZE: Final[int] = 256
THETA: Final[np.ndarray] = np.linspace(0, 180, 180, endpoint=False)
NUM_SAMPLES: Final[int] = 5  # Number of synthetic samples
NOISE_LEVEL: Final[float] = 0.1  # Gaussian noise standard deviation

# DICOM configuration
DICOM_METADATA: Final[dict] = {
    "PatientID": "ANONYMOUS",
    "PatientName": "CT Phantom",
    "Modality": "CT",
    "BitsAllocated": 16,
    "PixelRepresentation": 0,
}

###################################################################################

HOUNSFIELD_AIR: Final[int] = -1000
HOUNSFIELD_WATER: Final[int] = 0
HOUNSFIELD_BONE: Final[int] = 1000
HOUNSFIELD_SOFT_TISSUE: Final[tuple] = (20, 80)
