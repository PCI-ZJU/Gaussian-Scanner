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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2
import torch
import os
from typing import NamedTuple, Optional, List, Tuple
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        # import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    if cam_info.mask is not None:
        resized_mask = resize_mask_image(cam_info.mask, resolution)
        loaded_mask = torch.from_numpy(resized_mask).unsqueeze(0)
        # print(loaded_mask.max())
    else:
        loaded_mask = None

    ### load the depth of depth_anything 
    mono_depth_path_png = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "depth", cam_info.image_name+'.png')
    if os.path.exists(mono_depth_path_png):
        loaded_depth = load_raw_depth(mono_depth_path_png)
        resized_depth = cv2.resize(loaded_depth, resolution, interpolation=cv2.INTER_NEAREST)
        mono_depth = torch.from_numpy(resized_depth).unsqueeze(0)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, mono_depth=mono_depth,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def resize_mask_image(mask, resolution):
    width, height = resolution

    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return resized_mask

# def transform_poses_pca()

def bbox_from_mask_torch(mask : torch.tensor, margin=12, min_size=32):
    # mask.shape : [1, H, W]
    # return : bbox 
    if mask.ndim == 3:
        mask = mask[0]
    H, W = mask.shape
    m = (mask > 0.5)
    if m.sum() == 0:
        print(f'[Warning] An empty mask found, plz check the mask!')
        return 0, 0, W, H
    ys = torch.where(m.sum(dim=1) > 0)[0]
    xs = torch.where(m.sum(dim=0) > 0)[0]

    y0, y1 = int(ys[0]), int(ys[-1])
    x0, x1 = int(xs[0]), int(xs[-1])

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(W - 1, x1 + margin)
    y1 = min(H - 1, y1+ margin)

    w = max(min_size, x1 - x0 + 1)
    h = max(min_size, y1 - y0 + 1)

    if x0 + w > W:
        x0 = max(0, W - w)
    if y0 + h > H:
        y0 = max(0, H - h)
    return x0, y0, w, h
def load_raw_depth(fpath="raw.png"):
    depth = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    depth = depth.astype(np.float32) / 255.0
    # depth = (depth / 1000).astype(np.float32)
    return depth