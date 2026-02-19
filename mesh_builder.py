import numpy as np
import trimesh
import open3d as o3d
from skimage.measure import marching_cubes
from scipy.ndimage import binary_fill_holes, label, gaussian_filter, zoom
from tqdm import tqdm


def extract_body_mask(volume: np.ndarray, hu_threshold: float = -200) -> np.ndarray:
    
    body = (volume > hu_threshold).astype(np.uint8)

    for i in range(body.shape[0]):
        body[i] = binary_fill_holes(body[i])
    for i in range(body.shape[1]):
        body[:, i, :] = binary_fill_holes(body[:, i, :])
    for i in range(body.shape[2]):
        body[:, :, i] = binary_fill_holes(body[:, :, i])

    labeled, n = label(body)
    if n > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        body = (labeled == counts.argmax()).astype(np.uint8)

    return body


def build_body_mesh(
    volume: np.ndarray,
    spacing: np.ndarray,
    hu_threshold: float = -200,
    smooth_iterations: int = 15,
    target_faces: int = 300_000,      
    downsample: float = 0.5,         
) -> trimesh.Trimesh:
   
    spacing = np.array([float(s) for s in spacing])

    print(f"Downsampling ({downsample:.0%})...")
    vol_ds = zoom(volume.astype(np.float32), downsample, order=1).astype(np.int16)
    spacing_ds = spacing / downsample

    body_mask = extract_body_mask(vol_ds, hu_threshold)
    smooth = gaussian_filter(body_mask.astype(np.float32), sigma=1.2)
    
    # step_size=1 for higher accuracy
    verts, faces, normals, _ = marching_cubes(
        smooth, level=0.5, spacing=spacing_ds, step_size=1
    )

    del vol_ds, body_mask, smooth

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    
    o3d_mesh = o3d_mesh.filter_smooth_laplacian(
        number_of_iterations=smooth_iterations, lambda_filter=0.5
    )
    
    if len(o3d_mesh.triangles) > target_faces:
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
    
    o3d_mesh.compute_vertex_normals()

    mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_normals=np.asarray(o3d_mesh.vertex_normals),
    )
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    
    return mesh


def build_organ_mesh(
    mask: np.ndarray,
    spacing: np.ndarray,
    downsample: float = 0.5,
    smooth_iter: int = 2,  # Reduced from 5
) -> trimesh.Trimesh | None:
  
    if mask.sum() < 100:
        return None

    spacing = np.array([float(s) for s in spacing])

    if downsample < 1.0:
        mask_ds = zoom(mask.astype(np.float32), downsample, order=0) > 0.5
        spacing_ds = spacing / downsample
    else:
        mask_ds = mask
        spacing_ds = spacing

    # Lighter smoothing
    smooth = gaussian_filter(mask_ds.astype(np.float32), sigma=0.8)

    try:
        verts, faces, normals, _ = marching_cubes(
            smooth, level=0.5, spacing=spacing_ds, step_size=1
        )
    except Exception:
        return None

    if len(verts) < 10:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    

    if smooth_iter > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iter)
    
    trimesh.repair.fix_normals(mesh)
    return mesh


def build_all_organ_meshes(
    organ_masks: dict[str, np.ndarray],
    spacing: np.ndarray,
    downsample: float = 0.5,
    min_voxels: int = 500,
    smooth_iter: int = 2,
) -> dict[str, trimesh.Trimesh]:
   
    meshes = {}
    organs = {k: v for k, v in organ_masks.items() if v.sum() >= min_voxels}

    print(f"       Building {len(organs)} organs...")

    for name in tqdm(sorted(organs.keys()), desc="       Organs"):
        mesh = build_organ_mesh(
            organs[name], spacing, 
            downsample=downsample,
            smooth_iter=smooth_iter
        )
        if mesh is not None:
            meshes[name] = mesh

    return meshes