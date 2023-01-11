import torch
import torch.nn as nn
import numpy as np
from se3pose import OptimizablePose
from utils.sample_util import *

rays_dir = None

class RGBDFrame(nn.Module):
    def __init__(self, fid, rgb, depth, K, pose=None) -> None:
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        self.rgb = rgb.cuda() 
        self.depth = depth.cuda() #/ 2
        self.K = K
        # self.register_buffer("rgb", rgb)
        # self.register_buffer("depth", depth)

        if pose is not None:
            pose[:3, 3] += 10
            pose = torch.tensor(pose, requires_grad=True, dtype=torch.float32)
            self.pose = OptimizablePose.from_matrix(pose)
            self.optim = torch.optim.Adam(self.pose.parameters(), lr=1e-3)
        else:
            self.pose = None
            self.optim = None
        self.precompute()

    def get_pose(self):
        return self.pose.matrix()

    def get_translation(self):
        return self.pose.translation()

    def get_rotation(self):
        return self.pose.rotation()

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy')
        rays_d = torch.stack(
                    [(ix-K[0, 2]) / K[0,0],
                    (iy-K[1,2]) / K[1,1],
                    torch.ones_like(ix)], -1).float()
        return rays_d

    @torch.no_grad()
    def precompute(self):
        global rays_dir
        if rays_dir is None:
            rays_dir = self.get_rays(K=self.K).cuda()
        self.rays_d = rays_dir

    @torch.no_grad()
    def get_points(self):
        vmap = self.rays_d * self.depth[..., None]
        return vmap[self.depth > 0].reshape(-1, 3)

    @torch.no_grad()
    def sample_rays(self, N_rays):
        self.sample_mask = sample_rays(
            torch.ones_like(self.depth)[None, ...], N_rays)[0, ...]
