import numpy as np
import trimesh


def check_embree_available() -> bool:
    
    try:
        test_mesh = trimesh.creation.box()
        intersector = test_mesh.ray
        if 'embree' in str(type(intersector)).lower():
            return True
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return True
    except Exception:
        return False


EMBREE_AVAILABLE = check_embree_available()


class BodyProjector:
   
    
    def __init__(
        self, 
        body_mesh: trimesh.Trimesh,
        anatomical_directions: dict,
    ):
        self.body_mesh = body_mesh
        self.directions = anatomical_directions
        self.body_center = body_mesh.centroid
        self._ray = body_mesh.ray
        print(f"Ray backend: {type(self._ray).__name__}")
        
    def _intersect_rays(
        self, 
        origins: np.ndarray, 
        directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
       
        return self._ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions
        )
    
    def project_point_to_surface(
        self,
        point: np.ndarray,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
       
        closest, _, fid = trimesh.proximity.closest_point(self.body_mesh, [point])
        surf_pt = closest[0]
        surf_normal = self.body_mesh.face_normals[fid[0]]
        
        # Ensure normal points outward (away from body center)
        to_surface = surf_pt - self.body_center
        if np.dot(surf_normal, to_surface) < 0:
            surf_normal = -surf_normal
        
        # Apply offset
        final_pt = surf_pt + surf_normal * offset
        
        return final_pt, surf_normal
    
    def project_point_outward(
        self,
        point: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        locations, _, face_ids = self._intersect_rays(
            np.array([point]), 
            np.array([direction])
        )
        
        if len(locations) == 0:
            return None, None
        
        normals = self.body_mesh.face_normals[face_ids]
        dots = np.einsum('ij,j->i', normals, direction)
        outward_mask = dots > 0
        
        if not np.any(outward_mask):
            distances = np.linalg.norm(locations - point, axis=1)
            idx = np.argmax(distances)
            surf_pt = locations[idx]
            surf_normal = normals[idx]
            if np.dot(surf_normal, surf_pt - self.body_center) < 0:
                surf_normal = -surf_normal
            return surf_pt, surf_normal
        
        outward_locs = locations[outward_mask]
        outward_normals = normals[outward_mask]
        distances = np.linalg.norm(outward_locs - point, axis=1)
        idx = np.argmax(distances)
        
        return outward_locs[idx], outward_normals[idx]
    
    def get_organ_sample_points(
        self, 
        organ_mesh: trimesh.Trimesh,
        n_samples: int = 12
    ) -> np.ndarray:
    
        bounds = organ_mesh.bounds
        centroid = organ_mesh.centroid
        
        points = [centroid]
        
        for axis in range(3):
            for sign in [-1, 1]:
                pt = centroid.copy()
                pt[axis] = bounds[0 if sign < 0 else 1][axis]
                points.append(pt)
        
        if len(organ_mesh.vertices) > 20 and n_samples > 7:
            n_surface = min(n_samples - 7, 8)
            try:
                surface_pts, _ = trimesh.sample.sample_surface(organ_mesh, n_surface)
                points.extend(surface_pts)
            except Exception:
                pass
        
        return np.array(points)
    
    def estimate_radius_from_projection(
        self,
        organ_mesh: trimesh.Trimesh,
        projected_points: np.ndarray,
        surface_normal: np.ndarray,
    ) -> float:
   
        if len(projected_points) < 3:
            extents = organ_mesh.bounds[1] - organ_mesh.bounds[0]
            return np.mean(extents) / 3
        
        center = np.mean(projected_points, axis=0)
        
        n = surface_normal / (np.linalg.norm(surface_normal) + 1e-8)
        if abs(n[2]) < 0.9:
            u = np.cross(n, [0, 0, 1])
        else:
            u = np.cross(n, [0, 1, 0])
        u = u / (np.linalg.norm(u) + 1e-8)
        v = np.cross(n, u)
        
        relative = projected_points - center
        coords_2d = np.column_stack([np.dot(relative, u), np.dot(relative, v)])
        distances = np.linalg.norm(coords_2d, axis=1)
        
        return np.percentile(distances, 90) if len(distances) > 3 else np.max(distances)
    
    def project_organ(
        self,
        organ_mesh: trimesh.Trimesh,
        direction_name: str = "anterior",
        n_samples: int = 12,
    ) -> dict:
     
        direction = self.directions.get(direction_name, self.directions["anterior"])
        direction = np.array(direction, dtype=np.float64)
        
        sample_points = self.get_organ_sample_points(organ_mesh, n_samples)
        
        projected = []
        normals = []
        
        for pt in sample_points:
            proj_pt, proj_normal = self.project_point_outward(pt, direction)
            if proj_pt is not None:
                projected.append(proj_pt)
                normals.append(proj_normal)
        
        if not projected:
            return {
                "center": None,
                "normal": None,
                "radius": 0,
                "points": np.array([]),
                "valid": False,
            }
        
        projected = np.array(projected)
        normals = np.array(normals)
        
        center = np.mean(projected, axis=0)
        
        distances = np.linalg.norm(projected - center, axis=1)
        weights = 1 / (distances + 1)
        weights /= weights.sum()
        avg_normal = np.average(normals, axis=0, weights=weights)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
        
        radius = self.estimate_radius_from_projection(organ_mesh, projected, avg_normal)
        
        return {
            "center": center,
            "normal": avg_normal,
            "radius": radius,
            "points": projected,
            "organ_centroid": organ_mesh.centroid,
            "valid": True,
        }
    
    def project_all_organs(
        self,
        organ_meshes: dict[str, trimesh.Trimesh],
        direction: str = "anterior",
        n_samples: int = 12,
    ) -> dict[str, dict]:
      
        dir_vec = self.directions.get(direction, self.directions["anterior"])
        print(f"Direction ({direction}): {dir_vec}")
        
        projections = {}
        for name, mesh in organ_meshes.items():
            proj = self.project_organ(mesh, direction, n_samples)
            projections[name] = proj
        
        return projections


def create_surface_patch(
    projected_points: np.ndarray,
    normal: np.ndarray,
    body_mesh: trimesh.Trimesh,
    offset: float = 5.0,
    subdivisions: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
   
    if len(projected_points) < 3:
        return None, None
    
    body_center = body_mesh.centroid
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    center_3d = np.mean(projected_points, axis=0)
    
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(normal, u)
   
    relative = projected_points - center_3d
    boundary_2d = np.column_stack([
        np.dot(relative, u),
        np.dot(relative, v)
    ])
    
    try:
        from scipy.spatial import ConvexHull, Delaunay
        hull = ConvexHull(boundary_2d)
        hull_indices = hull.vertices
        boundary_2d_ordered = boundary_2d[hull_indices]
        boundary_3d_ordered = projected_points[hull_indices]
    except Exception:
        angles = np.arctan2(boundary_2d[:, 1], boundary_2d[:, 0])
        hull_indices = np.argsort(angles)
        boundary_2d_ordered = boundary_2d[hull_indices]
        boundary_3d_ordered = projected_points[hull_indices]
    
    n_boundary = len(boundary_2d_ordered)
    interior_2d = []
    interior_3d = []

    interior_2d.append([0.0, 0.0])
    interior_3d.append(center_3d)

    if subdivisions > 0:
        center_2d = np.array([0.0, 0.0])
        for ring in range(1, subdivisions + 1):
            t = ring / (subdivisions + 1)
            for i in range(n_boundary):

                pt_2d = center_2d + t * (boundary_2d_ordered[i] - center_2d)
                pt_3d = center_3d + t * (boundary_3d_ordered[i] - center_3d)
                interior_2d.append(pt_2d)
                interior_3d.append(pt_3d)

    all_2d = np.vstack([boundary_2d_ordered, np.array(interior_2d)])
    all_3d = np.vstack([boundary_3d_ordered, np.array(interior_3d)])
 
    try:
        tri = Delaunay(all_2d)
        triangles = tri.simplices
    except Exception:
        n_pts = len(all_3d)
        center_idx = n_boundary  
        triangles = []
        for i in range(n_boundary):
            next_i = (i + 1) % n_boundary
            triangles.append([center_idx, i, next_i])
        triangles = np.array(triangles)
    
    final_vertices = []
    for pt_3d in all_3d:
        closest, _, fid = trimesh.proximity.closest_point(body_mesh, [pt_3d])
        surf_pt = closest[0]
        surf_normal = body_mesh.face_normals[fid[0]]
       
        to_surface = surf_pt - body_center
        if np.dot(surf_normal, to_surface) < 0:
            surf_normal = -surf_normal
    
        final_pt = surf_pt + surf_normal * offset
        final_vertices.append(final_pt)
    
    return np.array(final_vertices), triangles

def create_projection_outline(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    body_mesh: trimesh.Trimesh,
    segments: int = 32,
    offset: float = 5.0,
) -> np.ndarray:
 
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    body_center = body_mesh.centroid
    
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(normal, u)
    
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    points = []
    
    for angle in angles:
        circle_pt = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
        closest, _, fid = trimesh.proximity.closest_point(body_mesh, [circle_pt])
        surf_pt = closest[0]
        surf_normal = body_mesh.face_normals[fid[0]]
      
        to_surface = surf_pt - body_center
        if np.dot(surf_normal, to_surface) < 0:
            surf_normal = -surf_normal
        
        points.append(surf_pt + surf_normal * offset)
    
    return np.array(points)


def create_convex_hull_outline(
    projected_points: np.ndarray,
    normal: np.ndarray,
    body_mesh: trimesh.Trimesh,
    offset: float = 5.0,
) -> np.ndarray:
   
    if len(projected_points) < 3:
        return projected_points
    
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    center = np.mean(projected_points, axis=0)
    body_center = body_mesh.centroid
    
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(normal, u)
    
    relative = projected_points - center
    coords_2d = np.column_stack([np.dot(relative, u), np.dot(relative, v)])
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords_2d)
        hull_indices = hull.vertices
    except Exception:
        angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
        hull_indices = np.argsort(angles)
    
    outline = []
    for idx in hull_indices:
        pt_3d = projected_points[idx]
        closest, _, fid = trimesh.proximity.closest_point(body_mesh, [pt_3d])
        surf_pt = closest[0]
        surf_normal = body_mesh.face_normals[fid[0]]
        
        to_surface = surf_pt - body_center
        if np.dot(surf_normal, to_surface) < 0:
            surf_normal = -surf_normal
        
        outline.append(surf_pt + surf_normal * offset)
    
    return np.array(outline)