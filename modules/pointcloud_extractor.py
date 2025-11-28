import open3d as o3d
from .FCGF import FCGFExtractor
import numpy as np

class PointCloudExtractor():
    def __init__(self, voxel_size, fcgf_model, device = "cuda"):
        self.voxel_size = voxel_size
        self.fcgf = FCGFExtractor(self.voxel_size, fcgf_model, device = device)

    def FPFH(self, pc):
        pc_down = pc.voxel_down_sample(self.voxel_size)
        pc_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = 2 * self.voxel_size, max_nn = 30))
        pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_down, o3d.geometry.KDTreeSearchParamHybrid(radius = 5 * self.voxel_size, max_nn = 100))
        return pc_down, np.asarray(pc_fpfh.data, np.float32).T

    def FCGF(self, pc):
        pc_down, descriptors = self.fcgf.extract(pc)
        return pc_down, descriptors.detach().cpu().numpy()