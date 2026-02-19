import tempfile
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk

class CTImage:
  
    def __init__(self, sitk_image: sitk.Image):
        self.image = sitk_image
        self.volume = sitk.GetArrayFromImage(sitk_image)
        
        # Extract orientation matrix from DICOM
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array(sitk_image.GetOrigin())
        
        self.direction = direction
        self.spacing = spacing[::-1]  
        self.spacing_xyz = spacing
        self.origin = origin
        
        # Build transformation matrices
        self._build_transforms()
        
    def _build_transforms(self):
       
        self.voxel_to_world = np.eye(4)
        self.voxel_to_world[:3, :3] = self.direction @ np.diag(self.spacing_xyz)
        self.voxel_to_world[:3, 3] = self.origin
        
        # World to voxel (inverse)
        self.world_to_voxel = np.linalg.inv(self.voxel_to_world)
        
    def get_anatomical_directions(self) -> dict:
       
        
        x_dir = self.direction[:, 0]  
        y_dir = self.direction[:, 1]  
        z_dir = self.direction[:, 2]  
        
        
        axes = {'x': x_dir, 'y': y_dir, 'z': z_dir}
        
        return {
            "anterior": -y_dir,     
            "posterior": y_dir,     
            "left": x_dir,           
            "right": -x_dir,        
            "superior": z_dir,       
            "inferior": -z_dir,     
            "direction_matrix": self.direction,
        }
    
    def voxel_to_world_point(self, voxel_zyx: np.ndarray) -> np.ndarray:
   
        voxel_xyz = voxel_zyx[::-1]
        pt = np.append(voxel_xyz, 1)
        world = self.voxel_to_world @ pt
        return world[:3]
    
    def world_to_voxel_point(self, world: np.ndarray) -> np.ndarray:
       
        pt = np.append(world, 1)
        voxel_xyz = self.world_to_voxel @ pt
        return voxel_xyz[:3][::-1]  

    def validate_ct_modality(self) -> dict:
       
        vol = self.volume
        
        info = {
            "valid": True,
            "warnings": [],
            "hu_min": float(vol.min()),
            "hu_max": float(vol.max()),
            "hu_mean": float(vol.mean()),
        }
        
       
        if vol.min() > -500:
            info["warnings"].append("Min HU > -500: may not be true CT or missing air")
            info["valid"] = False
        
        if vol.max() < 200:
            info["warnings"].append("Max HU < 200: may not have bone/contrast")
        
        if vol.max() > 4000:
            info["warnings"].append("Max HU > 4000: possible metal artifacts")
        
       
        air_voxels = np.sum(vol < -900)
        bone_voxels = np.sum(vol > 300)
        soft_tissue = np.sum((vol > -100) & (vol < 200))
        
        total = vol.size
        info["air_fraction"] = air_voxels / total
        info["bone_fraction"] = bone_voxels / total
        info["soft_tissue_fraction"] = soft_tissue / total
        
        if info["air_fraction"] < 0.01:
            info["warnings"].append("Very little air detected - check if CT")
        
        return info


def load_dicom_series(dicom_dir: str) -> CTImage:
    
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    series_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    image = reader.Execute()
    
    ct = CTImage(image)
    
  
    validation = ct.validate_ct_modality()
    if validation["warnings"]:
        print("  CT Validation warnings:")
        for w in validation["warnings"]:
            print(f"    - {w}")
    
    return ct


def get_torso_bounds(volume: np.ndarray, spacing: np.ndarray) -> tuple[int, int]:
   
    z_dim = volume.shape[0]
    
    body_threshold = -200
    areas = np.array([np.sum(volume[z] > body_threshold) for z in range(z_dim)])
    
    max_area = np.max(areas)
    threshold = max_area * 0.4
    valid = np.where(areas > threshold)[0]
    
    if len(valid) < 10:
        return int(z_dim * 0.15), int(z_dim * 0.85)
    
    margin = int(20 / spacing[0])
    z_start = max(0, valid[0] - margin)
    z_end = min(z_dim, valid[-1] + margin)
    
    return z_start, z_end


def crop_to_torso(ct: CTImage) -> tuple[np.ndarray, int, int]:
  
    z_start, z_end = get_torso_bounds(ct.volume, ct.spacing)
    print(f"       Torso: slices {z_start}-{z_end} of {ct.volume.shape[0]}")
    return ct.volume[z_start:z_end].copy(), z_start, z_end


def segment_all(
    dicom_dir: str, 
    ct: CTImage,
    z_start: int = 0, 
    z_end: int | None = None
) -> dict[str, np.ndarray]:
    from totalsegmentator.python_api import totalsegmentator

    tmp_out = tempfile.mkdtemp()
    masks = {}
    
    try:
        out_path = Path(tmp_out)
        
        totalsegmentator(
            dicom_dir,
            out_path,
            fast=True,
            nr_thr_saving=1,
        )

        for nii_file in sorted(out_path.glob("*.nii.gz")):
            organ_name = nii_file.stem.replace(".nii", "")
            
            try:
                mask_img = sitk.ReadImage(str(nii_file))
                
                # Resample to CT space
                mask_resampled = sitk.Resample(
                    mask_img,
                    ct.image,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    mask_img.GetPixelID(),
                )
                
                mask_array = sitk.GetArrayFromImage(mask_resampled) > 0
                
                if z_end is not None:
                    mask_array = mask_array[z_start:z_end]
                
                if mask_array.sum() > 100:
                    masks[organ_name] = mask_array
                    
            except Exception as e:
                print(f"Warning: {organ_name}: {e}")
                continue

        return masks
        
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)