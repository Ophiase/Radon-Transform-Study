from typing import Final
import numpy as np

IMAGE_SIZE: Final[int] = 256
THETA: Final[np.ndarray] = np.linspace(0, 180, 180, endpoint=False)
DICOM_METADATA: Final[dict] = {
    "PatientID": "ANONYMOUS",
    "PatientName": "CT Phantom",
    "Modality": "CT",
    "BitsAllocated": 16,
    "PixelRepresentation": 0,
}