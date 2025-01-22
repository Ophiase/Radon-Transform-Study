# dicom_io.py
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.filebase import DicomBytesIO
import numpy as np
from datetime import datetime
from constants import DICOM_METADATA

def _create_dicom_base(image: np.ndarray, is_sinogram: bool = False) -> FileDataset:
    """Create valid DICOM structure with required metadata"""
    # File meta info
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "SYNTHETIC_CT_1.0"

    # Create dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0"*128)
    
    # Set required DICOM attributes
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = DICOM_METADATA["Modality"]
    ds.PatientID = DICOM_METADATA["PatientID"]
    ds.PatientName = DICOM_METADATA["PatientName"]
    
    # Image parameters
    ds.Rows, ds.Columns = image.shape
    ds.BitsAllocated = DICOM_METADATA["BitsAllocated"]
    ds.BitsStored = DICOM_METADATA["BitsAllocated"]
    ds.HighBit = DICOM_METADATA["BitsAllocated"] - 1
    ds.PixelRepresentation = DICOM_METADATA["PixelRepresentation"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Normalize and convert pixel data
    image = (image - image.min()) / (image.max() - image.min()) * 65535
    ds.PixelData = image.astype(np.uint16).tobytes()
    
    # Add specific metadata
    if is_sinogram:
        ds.SeriesDescription = "Radon Transform Sinogram"
        ds.ImageType = ["DERIVED", "SECONDARY"]
    else:
        ds.SeriesDescription = "Synthetic CT Phantom"
        ds.ImageType = ["ORIGINAL", "PRIMARY"]
    
    # Add required dates
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = datetime.now().strftime('%H%M%S')
    ds.ContentDate = current_date
    ds.ContentTime = current_time
    ds.StudyDate = current_date
    ds.StudyTime = current_time
    ds.SeriesDate = current_date
    ds.SeriesTime = current_time
    
    return ds

def save_phantom_dicom(phantom: np.ndarray, filename: str) -> None:
    """Save phantom image as valid DICOM file"""
    ds = _create_dicom_base(phantom)
    ds.save_as(filename)

def save_sinogram_dicom(sinogram: np.ndarray, filename: str) -> None:
    """Save sinogram as valid DICOM file"""
    ds = _create_dicom_base(sinogram, is_sinogram=True)
    ds.save_as(filename)

def load_dicom(filename: str) -> np.ndarray:
    """Load DICOM file with proper transfer syntax handling"""
    ds = pydicom.dcmread(filename)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    image = ds.pixel_array.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min())