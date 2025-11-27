#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8, save_depth_colored
from functools import partial
import open3d as o3d
import trimesh
from tqdm.auto import trange
from pytorch3d.structures import Meshes
from pytorch3d.renderer import(
    MeshRasterizer, MeshRenderer,
    SoftSilhouetteShader, SoftPhongShader,
    RasterizationSettings, PerspectiveCameras, PointLights, TexturesVertex
)
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
device=torch.device("cuda:0")
charb = lambda x, eps=1e-3: torch.sqrt(x * x + eps * eps)

def laplacian_smooth_mesh(mesh, num_iterations=10):
    import copy
    mesh_0 = copy.deepcopy(mesh)
    mesh_taubin = mesh_0.filter_smooth_taubin(number_of_iterations=20)
    mesh_lap = mesh_taubin.filter_smooth_simple(number_of_iterations=10)
    return mesh_lap

class UnlitShader(nn.Module):
    def forward(self, fragments, meshes, **kwargs):
        tex = meshes.sample_textures(fragments)
        return tex[..., 0, :]



def _o3d_to_p3d(mesh_o3d, learn_color=False):
    verts = torch.tensor(np.asarray(mesh_o3d.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh_o3d.triangles), dtype=torch.int64, device=device)

    if hasattr(mesh_o3d, "vertex_colors") and len(mesh_o3d.vertex_colors) == len(mesh_o3d.vertices):
        cols = torch.tensor(np.asarray(mesh_o3d.vertex_colors), dtype=torch.float32,device=device).clamp(0,1)
    else:
        cols = torch.ones_like(verts) * 0.7  #set gray colors
    if learn_color:
        cols = torch.nn.Parameter(cols)
    textures = TexturesVertex(verts_features=cols[None])
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    return mesh, cols

def _p3d_to_o3d(mesh_p3d):
    """
    Pytorch3D Meshes -> Open3D TriangleMesh
    """
    v = mesh_p3d.verts_packed().detach().cpu().numpy()
    f = mesh_p3d.faces_packed().detach().cpu().numpy()
    o3 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(f)
    )
    if mesh_p3d.textures is not None and hasattr(mesh_p3d.textures, "verts_features_packed"):
        col = mesh_p3d.textures.verts_features_packed().detach().cpu().numpy()
        if col.shape[0] == v.shape[0]:
            o3.vertex_colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
    o3.compute_vertex_normals()
    return o3

def _cam_from_o3d_params(cam_o3d):
    """
    cam_o3d: open3d.camera.PinholeCameraParameters
    return Pytorch3D PerspectiveCameras
    """
    intr = cam_o3d.intrinsic
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()
    W, H = intr.width, intr.height

    RT_cv = torch.tensor(np.asarray(cam_o3d.extrinsic), dtype=torch.float32, device=device)
    Op = torch.diag(torch.tensor([-1., -1., 1., 1.], device=device))
    RT_p3d = Op @ RT_cv
    R_p3d = RT_p3d[:3, :3]
    T_p3d = RT_p3d[:3, 3]

    cams = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        in_ndc=False, R=R_p3d[None],T=T_p3d[None], device=device
    )
    return cams, (H, W)

def _make_renderers(image_size):
    rast = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-5,
        faces_per_pixel=8
    )
    rasterizer = MeshRasterizer(raster_settings=rast)
    renderer_sil = MeshRenderer(rasterizer=rasterizer, shader=SoftSilhouetteShader())
    renderer_rgb = MeshRenderer(rasterizer=rasterizer, shader=UnlitShader())
    return rasterizer, renderer_sil, renderer_rgb

def _init_color_affine(n_views, use_rgb, enable=False):
    if not (use_rgb and enable):
        return None, None
    s_list, b_list = [], []
    for _ in range(n_views):
        s = torch.nn.Parameters(torch.ones(3, device=device))
        b = torch.nn.Parameters(torch.ones(3, device=device))
        s_list.append(s)
        b_list.append(s)
        return s_list, b_list

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            # self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        # poses = c2ws[:,:3,:] @ np.diag([1, -1, 1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].mask is not None):
                depth[(self.viewpoint_stack[i].mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_depth_colored(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path,'depth_colored_{0:05d}'.format(idx)+".png"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

    
    def refine_mesh_photometric_no_depth(
            self,
            mesh_o3d,
            iters=1500,
            use_rgb=True,
            use_silhouette=True,
            use_2dgs_normals=False,
            learn_vertex_color=False,
            color_affine=True,
            # loss weight
            w_sil=1.0, w_rgb=0.7, w_norm_img=0.2,
            w_lap=0.2, w_norm=0.1, w_edge=0.05, w_anchor=0.03
    ):
        mesh_p3d, cols_param = _o3d_to_p3d(mesh_o3d, learn_color=learn_vertex_color)

        v_init = mesh_p3d.verts_packed().detach()
        verts = torch.nn.Parameter(v_init.clone())
        params = [verts]
        if learn_vertex_color and isinstance(cols_param, torch.nn.Parameter):
            params.append(cols_param)
        
        # set camera and rasterizer
        cams_o3d = list(to_cam_open3d(self.viewpoint_stack))
        cams0, (H0, W0) = _cam_from_o3d_params(cams_o3d[0])
        rasterizer, renderer_sil, renderer_rgb = _make_renderers((H0, W0))

        s_list, b_list = _init_color_affine(len(self.viewpoint_stack), use_rgb, color_affine)
        if s_list is not None:
            params += s_list + b_list
        
        with torch.no_grad():
            mesh_init = Meshes(verts=[v_init], faces=mesh_p3d.faces_list(), textures=mesh_p3d.textures)
            e = mesh_init.edges_packed()
            v = mesh_init.verts_packed()
            init_edge_len = (v[e[:,0]] - v[e[:, 1]]).norm(dim=-1).mean().item()
        
        opt = torch.optim.Adam(params, lr=1e-3)
        pbar = trange(iters, desc="Refine Progress", dynamic_ncols=True)
        for it in pbar:
            textures = mesh_p3d.textures if not learn_vertex_color else TexturesVertex(verts_features=cols_param[None])
            mesh_now = Meshes(verts=[verts], faces=mesh_p3d.faces_list(), textures=textures)

            total = 0.0
            for i, vp in enumerate(self.viewpoint_stack):
                cams_i, (Hi, Wi) = _cam_from_o3d_params(cams_o3d[i])
                if (Hi, Wi) != (H0, W0):
                    rast_i, rend_sil_i, rend_rgb_i = _make_renderers((Hi, Wi))
                else:
                    rast_i, rend_sil_i, rend_rgb_i = rasterizer, renderer_sil, renderer_rgb
                
                ## rasterize
                frags = rast_i(mesh_now, cameras=cams_i)
                valid = (frags.pix_to_face[..., 0] >= 0).squeeze(0)

                if use_silhouette:
                    sil = rend_sil_i(mesh_now, cameras=cams_i)
                    sil_pred = sil[..., 3].squeeze(0).clamp(1e-4, 1-1e-4)
                if use_rgb:
                    rgb_pred = rend_rgb_i(mesh_now, cameras=cams_i)[0, ..., :3]
                

                ## GT
                rgb_gt = vp.original_image[0:3].permute(1,2,0).to(device)
                if getattr(vp, "mask", None) is not None:
                    # m = torch.tensor(vp.mask, device=device).bool()
                    m = vp.mask.bool().to(device)
                    if m.ndim == 3: m = m[0]
                else:
                    m = valid
                
                # loss
                if use_silhouette:
                    total += w_sil * F.binary_cross_entropy(sil_pred, m.float())
                
                if use_rgb and m.any():
                    if s_list is not None:
                        s, b = s_list[i], b_list[i]

                        total += 1e-3*((s-1)**2).mean() + 1e-4 * (b ** 2).mean()
                        rgb_pred_cmp = (rgb_pred * s[None, None, :] + b[None, None, :]).clamp(0, 1)
                    else:
                        rgb_pred_cmp = rgb_pred

                    total += w_rgb * charb(rgb_pred_cmp[m] - rgb_gt[m]).mean()
                
            total += w_lap * mesh_laplacian_smoothing(mesh_now)
            total += w_norm * mesh_normal_consistency(mesh_now)
            if w_edge > 0:
                total += w_edge * mesh_edge_loss(mesh_now, target_length=float(init_edge_len))
            if w_anchor > 0:
                total += w_anchor * charb(verts - v_init).mean()
                
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            pbar.set_postfix(loss=f"{total.detach().item():.6f}")
        
        mesh_final = Meshes(
            verts=[verts.detach()],
            faces = mesh_p3d.faces_list(),
            textures=mesh_p3d.textures if not learn_vertex_color else TexturesVertex(verts_features=cols_param[None].detach())
        )
        return _p3d_to_o3d(mesh_final)
            