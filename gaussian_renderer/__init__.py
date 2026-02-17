#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ---- Mirror-3DGS extensions (Paper §II-B, §II-C) ----
# Added render modes: mirror_mask, mirror_viewpoint, depth.
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.mirror_utils import compute_mirror_camera


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, separate_sh=False, override_color=None,
           use_trained_exp=False,
           # ---- Mirror-3DGS options ----
           mirror_transform=None,
           render_mirror_mask=False,
           render_depth=False):
    """
    Render the scene, with optional Mirror-3DGS extensions.

    New keyword arguments
    ---------------------
    mirror_transform : (4,4) tensor or None
        If provided, render from the *mirrored* viewpoint (Paper §II-C).
        Mirror Gaussians are suppressed via opacity * (1 - m).
    render_mirror_mask : bool
        If True, an additional rasterisation pass produces the rendered
        mirror-probability mask M  (Paper Eq. 4).
    render_depth : bool
        If True, an additional rasterisation pass produces an expected
        depth map (for depth supervision, Paper Eq. 11).
    """

    # Screen-space gradient tensor
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # --- Camera parameters (possibly mirrored) ---
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if mirror_transform is not None:
        # [Mirror-3DGS] construct mirrored camera matrices
        viewmatrix, projmatrix, campos = compute_mirror_camera(
            viewpoint_camera, mirror_transform)
    else:
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # [Mirror-3DGS] When rendering from the mirrored viewpoint the mirror
    # surface itself must be transparent so we "see through" the plane.
    if mirror_transform is not None:
        opacity = opacity * (1.0 - pc.get_mirror)

    # Covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Colours / SH
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - campos.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # --- Main rasterisation ---
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D, means2D=means2D, dc=dc, shs=shs,
            colors_precomp=colors_precomp, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D, means2D=means2D, shs=shs,
            colors_precomp=colors_precomp, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)

    # Exposure correction (vanilla 3DGS feature, not used by Mirror-3DGS)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (torch.matmul(rendered_image.permute(1, 2, 0),
                                       exposure[:3, :3]).permute(2, 0, 1)
                          + exposure[:3, 3, None, None])

    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }

    # -----------------------------------------------------------------
    # [Mirror-3DGS] Extra pass: rendered mirror mask  (Paper Eq. 4)
    # -----------------------------------------------------------------
    if render_mirror_mask:
        # Alpha-composite mirror probability m_i the same way as colour.
        # Rasteriser expects 3-channel input; we replicate m to 3 channels
        # and use a black background so unpainted pixels = 0.
        mirror_colors = pc.get_mirror.repeat(1, 3)  # (N, 3)
        mask_bg = torch.zeros(3, device="cuda")

        mask_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=mask_bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=pipe.antialiasing,
        )
        mask_rasterizer = GaussianRasterizer(mask_settings)
        mirror_mask_3ch, _, _ = mask_rasterizer(
            means3D=means3D, means2D=means2D, shs=None,
            colors_precomp=mirror_colors,
            opacities=pc.get_opacity,   # use original opacity (not suppressed)
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
        # Take first channel → (1, H, W)
        out["mirror_mask"] = mirror_mask_3ch[:1].clamp(0, 1)

    # -----------------------------------------------------------------
    # [Mirror-3DGS] Extra pass: expected depth  (for Eq. 11)
    # -----------------------------------------------------------------
    if render_depth:
        # Compute per-Gaussian depth in camera space
        pts_hom = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)  # (N, 4)
        pts_cam = pts_hom @ viewmatrix  # row-vector convention, result (N, 4)
        depths = pts_cam[:, 2:3].clamp(min=0.001)  # (N, 1)
        depth_colors = depths.repeat(1, 3)  # (N, 3) — replicate for 3-ch rasteriser

        depth_bg = torch.zeros(3, device="cuda")
        depth_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=depth_bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=pc.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=pipe.antialiasing,
        )
        depth_rasterizer = GaussianRasterizer(depth_settings)
        depth_3ch, _, _ = depth_rasterizer(
            means3D=means3D, means2D=means2D, shs=None,
            colors_precomp=depth_colors,
            opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
        out["rendered_depth"] = depth_3ch[:1]  # (1, H, W)

    return out
