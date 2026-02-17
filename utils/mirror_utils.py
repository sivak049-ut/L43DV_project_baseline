#
# Mirror-3DGS utilities: RANSAC plane fitting, mirror transform, and mask loading.
# Faithful to Meng et al., "Mirror-3DGS: Incorporating Mirror Reflections
# into 3D Gaussian Splatting", arXiv 2404.01168v2.
#

import os
import random
import glob
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# RANSAC plane fitting  (Paper §II-D, improved RANSAC [18])
# ---------------------------------------------------------------------------

def ransac_plane_fit(points, threshold=0.01, max_iterations=1000):
    """
    Fit a plane to 3D points using RANSAC.

    Args:
        points: (N, 3) numpy array.
        threshold: inlier distance threshold.
        max_iterations: RANSAC iterations.

    Returns:
        plane_eq: [a, b, c, d] with ||(a,b,c)|| = 1, s.t. ax+by+cz+d = 0.
        inlier_ids: numpy array of inlier indices.
    """
    n_points = points.shape[0]
    best_eq = None
    best_inliers = np.array([], dtype=int)

    for _ in range(max_iterations):
        ids = random.sample(range(n_points), 3)
        p0, p1, p2 = points[ids[0]], points[ids[1]], points[ids[2]]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p0)

        distances = np.abs(points @ normal + d)
        inlier_ids = np.where(distances <= threshold)[0]

        if len(inlier_ids) > len(best_inliers):
            best_eq = [normal[0], normal[1], normal[2], d]
            best_inliers = inlier_ids

    return best_eq, best_inliers


# ---------------------------------------------------------------------------
# Mirror transform matrix  (Paper Eq. 8)
# ---------------------------------------------------------------------------

def build_mirror_transform(plane_eq):
    """
    Build the 4x4 Householder reflection matrix T_m from the plane equation.

    T_m is symmetric and involutory (T_m @ T_m = I), so T_m^{-1} = T_m.

    Args:
        plane_eq: [a, b, c, d] with unit normal (a, b, c).

    Returns:
        T_m: (4, 4) float32 torch tensor on CUDA.
    """
    a, b, c, d = plane_eq
    T_m = np.array([
        [1 - 2*a*a, -2*a*b,    -2*a*c,    -2*a*d],
        [-2*a*b,    1 - 2*b*b, -2*b*c,    -2*b*d],
        [-2*a*c,    -2*b*c,    1 - 2*c*c, -2*c*d],
        [0,         0,         0,          1     ],
    ], dtype=np.float32)
    return torch.from_numpy(T_m).cuda()


# ---------------------------------------------------------------------------
# Mirrored camera matrices  (Paper Eq. 7)
# ---------------------------------------------------------------------------

def compute_mirror_camera(viewpoint_camera, mirror_transform):
    """
    Compute mirrored camera matrices for the 3DGS rasteriser.

    In vanilla 3DGS the stored matrices use *transposed* convention:
        world_view_transform  = W2C^T
        projection_matrix     = Proj^T
        full_proj_transform   = W2C^T @ Proj^T  = (Proj @ W2C)^T

    The mirrored W2C is:  W2C_m = W2C @ T_m
    Because T_m is symmetric, (W2C @ T_m)^T = T_m^T @ W2C^T = T_m @ W2C^T.

    Returns:
        mirror_viewmatrix:  (4,4)  mirrored world_view_transform
        mirror_projmatrix:  (4,4)  mirrored full_proj_transform
        mirror_campos:      (3,)   mirrored camera centre in world space
    """
    viewmatrix = viewpoint_camera.world_view_transform       # W2C^T
    full_proj  = viewpoint_camera.full_proj_transform         # (Proj @ W2C)^T

    mirror_viewmatrix = mirror_transform @ viewmatrix         # T_m @ W2C^T
    mirror_projmatrix = mirror_transform @ full_proj          # T_m @ (Proj @ W2C)^T

    # Reflected camera position: T_m applied to original camera centre.
    cam_pos = viewpoint_camera.camera_center                  # (3,)
    cam_hom = torch.cat([cam_pos, torch.ones(1, device=cam_pos.device)])
    mirror_campos = (mirror_transform @ cam_hom)[:3]

    return mirror_viewmatrix, mirror_projmatrix, mirror_campos


# ---------------------------------------------------------------------------
# Mirror-mask I/O helpers
# ---------------------------------------------------------------------------

def load_mirror_masks(source_path, mask_dir_name="mirror_masks"):
    """
    Load binary mirror masks from ``<source_path>/<mask_dir_name>/``.

    The mask filenames must match the corresponding image filenames
    (extension may differ).  Masks are returned as a dict mapping
    *stem* (filename without extension) -> (H, W) float32 numpy array in [0,1].

    Falls back to common alternative directory names if the primary one
    is not found: ``masks``, ``mirror_mask``, ``mask``.
    """
    candidates = [mask_dir_name, "masks", "mirror_mask", "mask"]
    mask_root = None
    for name in candidates:
        p = os.path.join(source_path, name)
        if os.path.isdir(p):
            mask_root = p
            break

    if mask_root is None:
        raise FileNotFoundError(
            f"No mirror mask directory found in {source_path}. "
            f"Tried: {candidates}.  Place binary mask PNGs there."
        )

    masks = {}
    for fpath in sorted(glob.glob(os.path.join(mask_root, "*"))):
        stem = os.path.splitext(os.path.basename(fpath))[0]
        img = np.array(Image.open(fpath).convert("L"), dtype=np.float32) / 255.0
        masks[stem] = img  # (H, W), values in [0, 1]

    print(f"[Mirror-3DGS] Loaded {len(masks)} mirror masks from {mask_root}")
    return masks


def get_mask_for_camera(mirror_masks, viewpoint_cam, resolution):
    """
    Retrieve and resize the mirror mask for a given camera, returned as a
    (1, H, W) CUDA float tensor.
    """
    stem = os.path.splitext(viewpoint_cam.image_name)[0]
    if stem not in mirror_masks:
        # Try the full name as a fallback
        if viewpoint_cam.image_name in mirror_masks:
            stem = viewpoint_cam.image_name
        else:
            # No mask for this view -> return zeros
            return torch.zeros(1, int(viewpoint_cam.image_height),
                               int(viewpoint_cam.image_width), device="cuda")

    mask_np = mirror_masks[stem]
    # Resize to match the camera resolution (W, H)
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize(
        (int(viewpoint_cam.image_width), int(viewpoint_cam.image_height)),
        Image.NEAREST,
    )
    mask_t = torch.from_numpy(np.array(mask_pil, dtype=np.float32) / 255.0)
    return mask_t.unsqueeze(0).cuda()  # (1, H, W)
