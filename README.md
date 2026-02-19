# Organ-Locator-Using-CT-Scan
Detection of organ's Location using patient's CT scan. 
This project reconstructs patient-specific 3D anatomy from CT DICOM scans and computes anatomically aligned surface projections of internal organs onto the skin.

The pipeline integrates medical image processing, mesh reconstruction, and computational geometry to generate surface-conforming projection patches derived directly from segmented organ geometry.

## Methodology

1. DICOM Processing
  -Load CT using SimpleITK
  -Extract direction matrix for anatomical alignment
  -Validate Hounsfield Unit (HU) ranges

2. Organ Segmentation
  -Deep learning-based segmentation using TotalSegmentator
  -Binary mask extraction for selected torso organs

3. 3D Reconstruction
  -Body surface extracted using HU thresholding (> -200 HU)
  -Marching Cubes (step_size=1) for high-resolution mesh generation
  -Laplacian smoothing and mesh decimation

4. Surface Projection
  -Dense sampling of organ mesh surface
  -Ray–triangle intersection along DICOM-aligned anatomical directions
  -Selection of outward-facing intersections
  -Local tangent-plane parameterization
  -2D Delaunay triangulation
  -Reprojection and outward offset to produce a surface-conforming patch

## Key Features

1. DICOM orientation-aware (no axis assumptions)

2. Patient-specific geometry (no atlas registration)

3. Multi-point ray-based projection

4. Surface-conforming triangulated patches

5. GLB export for external visualization

6. Optional PyEmbree acceleration

## Dependencies

Python 3.10
numpy
trimesh
open3d
SimpleITK
scipy
scikit-image
tqdm
TotalSegmentator
pyembree (optional)

## Usage

### Set the DICOM directory in main-iv.py and run:

python main-iv.py


### Output:

Interactive 3D visualization

Exported body_with_organsv2.glb

## Applications

1. Surgical visualization

2. Preoperative surface localization

3. Anatomical education

4. Research in geometric medical projection

## Limitations

1. Assumes straight-line projection

2. Depends on segmentation quality

3. Not validated for clinical decision-making
