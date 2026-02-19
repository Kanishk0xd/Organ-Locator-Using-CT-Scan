import gc
from segmentor2 import load_dicom_series, segment_all, crop_to_torso
from mesh_builder2 import build_body_mesh, build_all_organ_meshes
from visualizer1 import (
    visualize,
    visualize_projections_only,
    export_combined,
)

DICOM_DIR = "Your CT data directory here"    #ADD YOUR DICOM FOLDER PATH HERE

EXPORT_PATH = "body_with_organsv.glb"        #Saves a 3d file with the body and all organs. Set to None to skip exporting.
SHOW_VIEWER = True


SHOW_INTERNAL_ORGANS = True
SHOW_SURFACE_PROJECTIONS = True
PROJECTION_DIRECTION = "anterior"  
USE_CONVEX_HULL_OUTLINE = True    
N_PROJECTION_SAMPLES = 12         


HU_THRESHOLD = -200
SMOOTH_ITERATIONS = 15
BODY_TARGET_FACES = 300_000       
BODY_DOWNSAMPLE = 0.5             
BODY_EXPORT_ALPHA = 25

ORGAN_DOWNSAMPLE = 0.5
ORGAN_SMOOTH_ITER = 2           
MIN_ORGAN_VOXELS = 500


TORSO_ORGANS = {
    "liver", "pancreas", "spleen",
    "kidney_left", "kidney_right",
    "stomach", "colon", "small_bowel",
    "duodenum", "urinary_bladder",
    "gallbladder", "heart",
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
}

SELECTED_ORGANS = None # Set to a list of organ names from TORSO_ORGANS to visualize/export only those organs, or None to include all.




def main():
    print("CT Organ Visualization Pipeline")
   
    print("\n[1/5] Loading DICOM...")
    ct = load_dicom_series(DICOM_DIR)
    print(f"Volume: {ct.volume.shape}")
    print(f"Spacing: {ct.spacing} mm")
    print(f"Origin: {ct.origin}")
    print(f"Direction matrix:\n{ct.direction}")
   
    anatomical_dirs = ct.get_anatomical_directions()
    print(f"Anterior direction: {anatomical_dirs['anterior']}")

    print("\n[2/5] Cropping to torso...")
    volume_cropped, z_start, z_end = crop_to_torso(ct)
    print(f"Cropped: {volume_cropped.shape}")

    print("\n[3/5] Segmenting organs...")
    organ_masks = segment_all(DICOM_DIR, ct, z_start=z_start, z_end=z_end)
    print(f"       Found {len(organ_masks)} organs")

    organ_masks = {k: v for k, v in organ_masks.items() if k in TORSO_ORGANS}
    print(f"Filtered to {len(organ_masks)} torso organs")

    gc.collect()

    print("\n[4/5] Building meshes...")
    
    print("Body surface (high accuracy)...")
    body_mesh = build_body_mesh(
        volume_cropped,
        ct.spacing,
        hu_threshold=HU_THRESHOLD,
        smooth_iterations=SMOOTH_ITERATIONS,
        target_faces=BODY_TARGET_FACES,
        downsample=BODY_DOWNSAMPLE,
    )
    print(f"Body: {len(body_mesh.vertices)} verts, {len(body_mesh.faces)} faces")

    print("Organ meshes (minimal smoothing)...")
    organ_meshes = build_all_organ_meshes(
        organ_masks,
        ct.spacing,
        downsample=ORGAN_DOWNSAMPLE,
        min_voxels=MIN_ORGAN_VOXELS,
        smooth_iter=ORGAN_SMOOTH_ITER,
    )
    print(f"Built {len(organ_meshes)} organ meshes")

    del organ_masks, volume_cropped
    gc.collect()

    print("\n[5/5] Output...")

    if EXPORT_PATH:
        export_combined(
            body_mesh,
            organ_meshes,
            EXPORT_PATH,
            body_opacity=BODY_EXPORT_ALPHA,
            selected_organs=SELECTED_ORGANS,
        )

    if SHOW_VIEWER:
        if SHOW_INTERNAL_ORGANS:
            visualize(
                body_mesh,
                organ_meshes,
                anatomical_dirs,
                selected_organs=SELECTED_ORGANS,
                show_organs=True,
                show_projections=SHOW_SURFACE_PROJECTIONS,
                projection_direction=PROJECTION_DIRECTION,
                use_convex_hull=USE_CONVEX_HULL_OUTLINE,
                fill_projections=True,
                n_projection_samples=N_PROJECTION_SAMPLES,
            )
        else:
            visualize_projections_only(
                body_mesh,
                organ_meshes,
                anatomical_dirs,
                selected_organs=SELECTED_ORGANS,
                projection_direction=PROJECTION_DIRECTION,
            )

    
    print("Done!")

if __name__ == "__main__":
    main()