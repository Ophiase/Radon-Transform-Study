import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.filebase import DicomBytesIO
import numpy as np
from datetime import datetime
from constants import DICOM_METADATA

def _create_dicom_base(image: np.ndarray, is_sinogram: bool = False) -> FileDataset:
    """Create DICOM dataset compliant with CT Image Storage SOP Class"""
    # File Meta Information
    file_meta = pydicom.FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    
    # Main Dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0"*128)
    
    # Patient Module
    ds.PatientID = DICOM_METADATA["PatientID"]
    ds.PatientName = DICOM_METADATA["PatientName"]
    
    # Study Module
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    
    # Series Module
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.Modality = "CT"
    ds.SeriesDate = ds.StudyDate
    ds.SeriesTime = ds.StudyTime
    
    # Frame of Reference
    ds.FrameOfReferenceUID = generate_uid()
    
    # Image Pixel Module
    ds.Rows, ds.Columns = image.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # CT Image Module
    ds.KVP = 120  # Tube voltage in kV
    ds.ExposureTime = 1000  # In ms
    ds.XRayTubeCurrent = 300  # In mA
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1000.0
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    ds.SliceThickness = 1.0
    
    # Convert to Hounsfield Units
    hu_image = _to_hounsfield(image)  
    ds.PixelData = hu_image.astype(np.int16).tobytes()
    
    # Type-specific metadata
    if is_sinogram:
        ds.SeriesDescription = "Radon Transform Sinogram"
        ds.ImageType = ["DERIVED", "SECONDARY"]
        ds.BodyPartExamined = "SYNTHETIC"
    else:
        ds.SeriesDescription = "Synthetic CT Phantom"
        ds.ImageType = ["ORIGINAL", "PRIMARY"] 
        ds.BodyPartExamined = "ABDOMEN"
    
    return ds


def _to_hounsfield(tensor: np.ndarray) -> np.ndarray:
    # Scale [0,1] -> [-1000, +1000] HU
    return (tensor * 2000) - 1000

def _from_hounsfield(tensor: np.ndarray) -> np.ndarray:
    # [-1000, +1000] HU -> Scale [0,1]  
    # return np.clip((tensor + 1000) / 2000, 0.0, 1.0) # not working?
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

###################################################################################


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
    return _from_hounsfield(image)
