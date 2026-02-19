import numpy as np
import open3d as o3d
import trimesh
from projector1 import (
    BodyProjector,
    create_projection_outline,
    create_convex_hull_outline,
    create_surface_patch,
)

ORGAN_COLORS = {
    "spleen": [0.6, 0.2, 0.6],
    "liver": [0.7, 0.3, 0.1],
    "kidney_right": [0.9, 0.4, 0.1],
    "kidney_left": [0.9, 0.5, 0.1],
    "gallbladder": [0.2, 0.7, 0.3],
    "stomach": [1.0, 0.6, 0.7],
    "pancreas": [1.0, 0.9, 0.2],
    "small_bowel": [0.9, 0.7, 0.5],
    "duodenum": [0.85, 0.65, 0.45],
    "colon": [0.8, 0.5, 0.3],
    "urinary_bladder": [0.3, 0.5, 0.9],
    "heart": [0.9, 0.1, 0.1],
    "aorta": [0.8, 0.0, 0.0],
    "lung_upper_lobe_left": [0.3, 0.6, 0.9],
    "lung_lower_lobe_left": [0.25, 0.5, 0.85],
    "lung_upper_lobe_right": [0.3, 0.6, 0.9],
    "lung_middle_lobe_right": [0.28, 0.55, 0.88],
    "lung_lower_lobe_right": [0.25, 0.5, 0.85],
}


def get_organ_color(name: str) -> list:
    if name in ORGAN_COLORS:
        return ORGAN_COLORS[name]
    if "lung" in name:
        return [0.3, 0.6, 0.9]
    return [0.6, 0.6, 0.6]


def trimesh_to_o3d(mesh: trimesh.Trimesh, color=None) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.compute_vertex_normals()
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


def create_line_loop(points: np.ndarray, color: list) -> o3d.geometry.LineSet:
    n = len(points)
    if n < 2:
        return None
    lines = [[i, (i + 1) % n] for i in range(n)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def create_marker(center: np.ndarray, size: float, color: list) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def create_surface_conforming_patch(
    vertices: np.ndarray,
    triangles: np.ndarray,
    color: list,
) -> o3d.geometry.TriangleMesh | None:
   
    if vertices is None or triangles is None:
        return None
    
    if len(vertices) < 3 or len(triangles) < 1:
        return None
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    return mesh


def visualize(
    body_mesh: trimesh.Trimesh,
    organ_meshes: dict[str, trimesh.Trimesh],
    anatomical_directions: dict,
    selected_organs: list[str] | None = None,
    show_axes: bool = True,
    show_organs: bool = True,
    show_projections: bool = False,
    projection_direction: str = "anterior",
    use_convex_hull: bool = True,
    marker_size: float = 4.0,
    outline_offset: float = 5.0,
    n_projection_samples: int = 16,
    fill_projections: bool = True,
    show_outline: bool = True,
    show_center_marker: bool = False,
    patch_subdivisions: int = 2,  # More subdivisions = better surface conforming
):
  
    geometries = []

    # Organs
    if show_organs:
        for name, mesh in sorted(organ_meshes.items()):
            if selected_organs and name not in selected_organs:
                continue
            geometries.append(trimesh_to_o3d(mesh, get_organ_color(name)))

    # Projections
    if show_projections:
        print("\n  Computing surface-conforming projections...")
        projector = BodyProjector(body_mesh, anatomical_directions)
        projections = projector.project_all_organs(
            organ_meshes,
            projection_direction,
            n_samples=n_projection_samples
        )
        
        for name, data in projections.items():
            if selected_organs and name not in selected_organs:
                continue
            if not data["valid"]:
                continue
            
            color = get_organ_color(name)
            center = data["center"]
            normal = data["normal"]
            radius = data["radius"]
            proj_pts = data["points"]
            
            # Create filled surface-conforming patch
            if fill_projections and len(proj_pts) >= 3:
                vertices, triangles = create_surface_patch(
                    proj_pts,
                    normal,
                    body_mesh,
                    offset=outline_offset,
                    subdivisions=patch_subdivisions,
                )
                
                patch = create_surface_conforming_patch(vertices, triangles, color)
                if patch:
                    geometries.append(patch)
            
            # Create outline
            if show_outline:
                if use_convex_hull and len(proj_pts) >= 3:
                    outline = create_convex_hull_outline(
                        proj_pts, normal, body_mesh, outline_offset
                    )
                else:
                    outline = create_projection_outline(
                        center, radius, normal, body_mesh,
                        segments=32, offset=outline_offset
                    )
                
                line_set = create_line_loop(outline, [0, 0, 0])  # Black outline
                if line_set:
                    geometries.append(line_set)
            
            # Center marker
            if show_center_marker:
                # Project center to surface too
                closest, _, fid = trimesh.proximity.closest_point(body_mesh, [center])
                surf_normal = body_mesh.face_normals[fid[0]]
                to_surface = closest[0] - body_mesh.centroid
                if np.dot(surf_normal, to_surface) < 0:
                    surf_normal = -surf_normal
                marker_pos = closest[0] + surf_normal * (outline_offset + 2)
                geometries.append(create_marker(marker_pos, marker_size, color))
            
            print(f"    {name}: {len(proj_pts)} pts, r={radius:.1f}mm")

    # Body
    geometries.append(trimesh_to_o3d(body_mesh, [0.85, 0.85, 0.9]))

    # Axes
    if show_axes:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=80))

    print(f"\n  Display: {len(organ_meshes)} organs")
    print("  Controls: rotate=mouse, zoom=scroll, quit=Q\n")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="CT Visualization",
        width=1400, height=900,
        mesh_show_back_face=True,
    )


def visualize_projections_only(
    body_mesh: trimesh.Trimesh,
    organ_meshes: dict[str, trimesh.Trimesh],
    anatomical_directions: dict,
    selected_organs: list[str] | None = None,
    projection_direction: str = "anterior",
    fill_projections: bool = True,
    outline_offset: float = 5.0,
):
    
    visualize(
        body_mesh, organ_meshes, anatomical_directions,
        selected_organs=selected_organs,
        show_axes=False,
        show_organs=False,
        show_projections=True,
        projection_direction=projection_direction,
        fill_projections=fill_projections,
        show_outline=True,
        show_center_marker=False,
        outline_offset=outline_offset,
    )


def export_combined(
    body_mesh: trimesh.Trimesh,
    organ_meshes: dict[str, trimesh.Trimesh],
    output_path: str,
    body_opacity: int = 40,
    selected_organs: list[str] | None = None,
):
    scene = trimesh.Scene()

    body_copy = body_mesh.copy()
    body_copy.visual.face_colors = np.full(
        (len(body_copy.faces), 4), [220, 215, 210, body_opacity], dtype=np.uint8
    )
    scene.add_geometry(body_copy, node_name="body")

    for name, mesh in sorted(organ_meshes.items()):
        if selected_organs and name not in selected_organs:
            continue
        color = [int(c * 255) for c in get_organ_color(name)] + [255]
        organ_copy = mesh.copy()
        organ_copy.visual.face_colors = np.full(
            (len(organ_copy.faces), 4), color, dtype=np.uint8
        )
        scene.add_geometry(organ_copy, node_name=name)

    scene.export(output_path)
    print(f"  Exported -> {output_path}")