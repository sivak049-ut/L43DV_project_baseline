#
# Mirror-NeRF dataset reader for vanilla 3DGS.
#
# Reads the transforms_train.json / transforms_test.json format used by
# the Mirror-NeRF dataset directly, without COLMAP conversion.
#
# Key differences from the Blender reader in vanilla 3DGS:
#   - Uses fx/fy/cx/cy intrinsics (not camera_angle_x)
#   - Supports per-frame intrinsics matrices
#   - JPG images (no RGBA alpha compositing)
#   - No axis flip (data uses OpenCV convention, not Blender)
#   - Generates seed point cloud from camera frustums
#

import os
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path

from scene.dataset_readers import (
    CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly,
)
from utils.graphics_utils import focal2fov, BasicPointCloud
from utils.sh_utils import SH2RGB


def readMirrorNeRFCameras(path, transformsfile, depths_folder, is_test):
    """
    Read cameras from a Mirror-NeRF transforms JSON.

    The JSON has:
      - Global intrinsics: fx, fy, cx, cy  (optional fallback)
      - Per-frame: file_path, transform_matrix (4x4 c2w), intrinsics (3x3 K)

    CameraInfo convention (matching vanilla 3DGS):
      - R = R_c2w  (rotation part of camera-to-world, stored for later transpose)
      - T = t_w2c  (translation part of world-to-camera)
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as f:
        data = json.load(f)

    # Global intrinsics (fallback)
    global_fx = data.get("fx", None)
    global_fy = data.get("fy", None)
    global_cx = data.get("cx", None)
    global_cy = data.get("cy", None)

    frames = data["frames"]

    for idx, frame in enumerate(frames):
        # --- File path ---
        file_path = frame["file_path"]
        # Handle both "images/frame_00001.jpg" and "frame_00001.jpg"
        if os.path.isabs(file_path):
            image_path = file_path
        else:
            image_path = os.path.join(path, file_path)

        if not os.path.exists(image_path):
            # Try stripping the leading directory
            image_path = os.path.join(path, "images", os.path.basename(file_path))

        image_name = os.path.basename(file_path)
        image_stem = Path(image_name).stem

        # Read image to get dimensions
        img = Image.open(image_path)
        W, H = img.size

        # --- Intrinsics ---
        if "intrinsics" in frame:
            K = np.array(frame["intrinsics"])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        elif global_fx is not None:
            fx, fy = global_fx, global_fy
            cx, cy = global_cx, global_cy
        else:
            raise ValueError(
                f"No intrinsics found for frame {idx}. "
                f"Need either per-frame 'intrinsics' or global 'fx'/'fy'."
            )

        FovX = focal2fov(fx, W)
        FovY = focal2fov(fy, H)

        # --- Extrinsics ---
        # transform_matrix is camera-to-world (c2w)
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)

        # NOTE: Mirror-NeRF uses OpenCV convention (Y-down, Z-forward).
        # No axis flip is needed (unlike Blender format which needs c2w[:3,1:3]*=-1).
        # If your dataset uses Blender convention, uncomment the next line:
        # c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)

        # 3DGS CameraInfo convention:
        #   R = transpose of w2c rotation = R_c2w
        #   T = w2c translation
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        # --- Depth path ---
        depth_path = ""
        if depths_folder:
            candidate = os.path.join(depths_folder, f"{image_stem}.png")
            if os.path.exists(candidate):
                depth_path = candidate

        cam_infos.append(CameraInfo(
            uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
            depth_params=None,
            image_path=image_path, image_name=image_name,
            depth_path=depth_path,
            width=W, height=H,
            is_test=is_test,
        ))

    return cam_infos


def generatePointCloudFromCameras(cam_infos, num_random=10_000):
    """
    Generate a seed point cloud for 3DGS initialisation from camera positions
    plus random samples in the scene bounding volume.
    """
    from utils.graphics_utils import getWorld2View2

    cam_centers = []
    for cam in cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])

    cam_centers = np.array(cam_centers)
    center = cam_centers.mean(axis=0)
    extent = (cam_centers.max(axis=0) - cam_centers.min(axis=0)).max()

    rng = np.random.default_rng(42)
    random_pts = center + (rng.random((num_random, 3)) - 0.5) * extent * 1.5
    xyz = np.vstack([cam_centers, random_pts])
    colors = rng.random((len(xyz), 3)) * 0.5 + 0.25  # mild grey-ish
    return xyz, (colors * 255).astype(np.uint8)


def readMirrorNeRFSceneInfo(path, white_background, depths, eval, **kwargs):
    """
    Top-level reader for Mirror-NeRF datasets.

    Expects:
        path/transforms_train.json
        path/transforms_test.json   (optional)
        path/images/
        path/masks/ or path/mirror_masks/
    """
    depths_folder = os.path.join(path, depths) if depths else ""

    # --- Train cameras ---
    train_json = "transforms_train.json"
    if not os.path.exists(os.path.join(path, train_json)):
        train_json = "transforms.json"  # fallback
    print(f"Reading training cameras from {train_json}")
    train_cam_infos = readMirrorNeRFCameras(path, train_json, depths_folder,
                                            is_test=False)
    print(f"  {len(train_cam_infos)} training cameras loaded.")

    # --- Test cameras ---
    test_cam_infos = []
    test_json = "transforms_test.json"
    if os.path.exists(os.path.join(path, test_json)):
        print(f"Reading test cameras from {test_json}")
        test_cam_infos = readMirrorNeRFCameras(path, test_json, depths_folder,
                                               is_test=True)
        print(f"  {len(test_cam_infos)} test cameras loaded.")
    else:
        print("  No transforms_test.json found; test set will be empty.")

    if not eval:
        # When not in eval mode, merge test into train (vanilla 3DGS behaviour)
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    # --- Normalisation ---
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # --- Point cloud ---
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print("Generating seed point cloud from camera positions ...")
        xyz, rgb = generatePointCloudFromCameras(train_cam_infos)
        storePly(ply_path, xyz, rgb)
        print(f"  Saved {len(xyz)} points to {ply_path}")

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,  # JPG images, no alpha compositing
    )
    return scene_info


def is_mirrornerf_dataset(path):
    """
    Detect whether the given path contains a Mirror-NeRF format dataset.

    Heuristic: has transforms_train.json (or transforms.json) AND either
    a 'masks' or 'mirror_masks' directory, AND the JSON contains 'fx'
    or per-frame 'intrinsics' (distinguishing it from Blender format).
    """
    has_masks = (os.path.isdir(os.path.join(path, "masks")) or
                 os.path.isdir(os.path.join(path, "mirror_masks")))

    for json_name in ["transforms_train.json", "transforms.json"]:
        json_path = os.path.join(path, json_name)
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                # Mirror-NeRF uses fx/fy or per-frame intrinsics;
                # Blender uses camera_angle_x
                has_focal = "fx" in data or "fy" in data
                has_per_frame_K = (len(data.get("frames", [])) > 0 and
                                   "intrinsics" in data["frames"][0])
                if (has_focal or has_per_frame_K) and has_masks:
                    return True
            except:
                pass
    return False
