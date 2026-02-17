#!/usr/bin/env python3
"""
Mirror-3DGS: Render and evaluate on test views.

For each test camera this script:
  1. Renders from the original viewpoint
  2. Renders the mirror mask M
  3. Renders from the mirrored viewpoint
  4. Fuses: C_fuse = C_orig * (1-M) + C_mirror * M   (Paper Eq. 9)
  5. Saves the fused render and computes metrics against GT

Usage:
    python render_mirror.py \
        -m output/<run_id> \
        -s data/washroom \
        --eval

Outputs are written to:
    output/<run_id>/renders_mirror/     (fused rendered images)
    output/<run_id>/gt_test/            (ground-truth test images)
    output/<run_id>/renders_mask/       (rendered mirror masks)
    output/<run_id>/metrics.json        (PSNR, SSIM, LPIPS per image + mean)
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

try:
    from lpipsPyTorch import lpips as compute_lpips
    LPIPS_AVAILABLE = True
except:
    LPIPS_AVAILABLE = False
    print("WARNING: lpips not available. Install with: pip install lpips")


def render_test_views(dataset, pipe, iteration, mirror_plane_path=None):
    """
    Render all test views with mirror-aware fusion and compute metrics.
    """
    # ---- Load model ----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- Recover mirror plane ----
    # Option 1: Load from checkpoint (if plane was saved)
    mirror_transform = None
    plane_file = os.path.join(dataset.model_path, "mirror_plane.json")
    if os.path.exists(plane_file):
        with open(plane_file) as f:
            plane_data = json.load(f)
        gaussians.mirror_plane_eq = plane_data["plane_eq"]
        from utils.mirror_utils import build_mirror_transform
        mirror_transform = build_mirror_transform(gaussians.mirror_plane_eq)
        gaussians.mirror_transform = mirror_transform
        print(f"Loaded mirror plane from {plane_file}")
    else:
        # Option 2: Re-estimate from the trained Gaussians
        print("No saved mirror plane found — re-estimating from trained Gaussians ...")
        try:
            mirror_transform = gaussians.compute_mirror_plane(
                ransac_threshold=0.01, min_mirror_prob=0.3, min_opacity=0.3)
            # Save for future use
            plane_data = {"plane_eq": gaussians.mirror_plane_eq}
            with open(plane_file, "w") as f:
                json.dump(plane_data, f, indent=2)
            print(f"Saved mirror plane to {plane_file}")
        except RuntimeError as e:
            print(f"WARNING: Could not estimate mirror plane: {e}")
            print("  Rendering without mirror fusion (original viewpoint only).")

    # ---- Output directories ----
    render_dir = os.path.join(dataset.model_path, "renders_mirror")
    gt_dir = os.path.join(dataset.model_path, "gt_test")
    mask_dir = os.path.join(dataset.model_path, "renders_mask")
    orig_dir = os.path.join(dataset.model_path, "renders_original")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)

    # ---- Render loop ----
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("ERROR: No test cameras found. Did you pass --eval ?")
        return

    print(f"\nRendering {len(test_cameras)} test views ...")

    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_l1 = []
    per_image_metrics = {}

    for idx, viewpoint_cam in enumerate(tqdm(test_cameras, desc="Rendering")):
        # Ground truth
        gt_image = viewpoint_cam.original_image.cuda().clamp(0, 1)

        # Render original viewpoint + mirror mask
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                            render_mirror_mask=True)
        image_orig = render_pkg["render"].clamp(0, 1)
        rendered_mask = render_pkg.get("mirror_mask", None)

        # Mirror-fused rendering
        if mirror_transform is not None and rendered_mask is not None:
            # Render from mirrored viewpoint
            mirror_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                mirror_transform=mirror_transform)
            image_mirror = mirror_pkg["render"].clamp(0, 1)

            # Fuse  (Paper Eq. 9)
            M = rendered_mask.expand_as(image_orig)
            image_fused = image_orig * (1.0 - M) + image_mirror * M
        else:
            image_fused = image_orig
            rendered_mask = torch.zeros(1, int(viewpoint_cam.image_height),
                                        int(viewpoint_cam.image_width),
                                        device="cuda")

        image_fused = image_fused.clamp(0, 1)

        # ---- Compute metrics ----
        l1_val = l1_loss(image_fused, gt_image).item()
        psnr_val = psnr(image_fused, gt_image).mean().item()
        ssim_val = ssim(image_fused, gt_image).item()

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_l1.append(l1_val)

        lpips_val = None
        if LPIPS_AVAILABLE:
            lpips_val = compute_lpips(image_fused[None], gt_image[None],
                                      net_type='vgg', version='0.1').item()
            all_lpips.append(lpips_val)

        name = viewpoint_cam.image_name
        per_image_metrics[name] = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "l1": l1_val,
        }
        if lpips_val is not None:
            per_image_metrics[name]["lpips"] = lpips_val

        # ---- Save images ----
        def tensor_to_pil(t):
            return Image.fromarray(
                (t.detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255).astype(np.uint8))

        stem = os.path.splitext(name)[0]
        tensor_to_pil(image_fused).save(
            os.path.join(render_dir, f"{stem}.png"))
        tensor_to_pil(gt_image).save(
            os.path.join(gt_dir, f"{stem}.png"))
        tensor_to_pil(image_orig).save(
            os.path.join(orig_dir, f"{stem}.png"))

        # Save mask as grayscale
        mask_np = (rendered_mask[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(mask_np, mode="L").save(
            os.path.join(mask_dir, f"{stem}.png"))

    # ---- Aggregate metrics ----
    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)
    mean_l1 = np.mean(all_l1)

    metrics = {
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_l1": mean_l1,
        "num_test_views": len(test_cameras),
        "per_image": per_image_metrics,
    }

    if LPIPS_AVAILABLE and len(all_lpips) > 0:
        mean_lpips = np.mean(all_lpips)
        metrics["mean_lpips"] = mean_lpips
    else:
        mean_lpips = None

    # Save metrics
    metrics_path = os.path.join(dataset.model_path, "metrics_mirror.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*50}")
    print(f"  Mirror-3DGS Evaluation Results")
    print(f"{'='*50}")
    print(f"  Test views:  {len(test_cameras)}")
    print(f"  PSNR:        {mean_psnr:.4f}")
    print(f"  SSIM:        {mean_ssim:.6f}")
    print(f"  L1:          {mean_l1:.6f}")
    if mean_lpips is not None:
        print(f"  LPIPS:       {mean_lpips:.6f}")
    print(f"{'='*50}")
    print(f"  Renders:     {render_dir}")
    print(f"  GT:          {gt_dir}")
    print(f"  Masks:       {mask_dir}")
    print(f"  Metrics:     {metrics_path}")
    print(f"{'='*50}")

    return metrics


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Mirror-3DGS Render & Evaluate")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration to load (-1 = latest)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    # Ensure eval mode is on (needed for train/test split)
    args.eval = True

    print("Rendering " + args.model_path)

    dataset = model.extract(args)
    # Point model_path at the output directory
    dataset.model_path = args.model_path

    pipe = pipeline.extract(args)

    render_test_views(dataset, pipe, args.iteration)

    print("\nEvaluation complete.")
