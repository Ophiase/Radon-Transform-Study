import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import numpy as np
from datetime import datetime

from constants import DICOM_METADATA

def save_sinogram_dicom(sinogram: np.ndarray, filename: str) -> None:
    """Save sinogram as DICOM file with basic metadata"""
    ds = FileDataset(filename, {}, preamble=b"\0"*128)
    
    # Normalize and convert to uint16
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min()) * 65535
    sinogram = sinogram.astype(np.uint16)
    
    # Populate metadata
    ds.PatientID = DICOM_METADATA["PatientID"]
    ds.PatientName = DICOM_METADATA["PatientName"]
    ds.Modality = DICOM_METADATA["Modality"]
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    ds.BitsAllocated = DICOM_METADATA["BitsAllocated"]
    ds.PixelRepresentation = DICOM_METADATA["PixelRepresentation"]
    ds.Rows, ds.Columns = sinogram.shape
    ds.PixelData = sinogram.tobytes()
    
    ds.save_as(filename)