import torch
import numpy as np
import open3d as o3d
from .misc import extract_features
from .resunet import ResUNetBN2C

class FCGFExtractor:
    def __init__(self, voxel_size, model, device = "cuda"):
        self.device = device

        if model == "3DMatch":
            url = "https://node1.chrischoy.org/data/publications/fcgf/2019-08-19_06-17-41.pth"
            fname = "ResUNetBN2C-32feat-3conv.pth"
            print("FCGF with 3DMatch")
        elif model == "KITTI":
            url = "https://node1.chrischoy.org/data/publications/fcgf/2019-07-31_19-37-00.pth"
            fname = "KITTI-v0.3-ResUNetBN2C-3conv-nout32.pth"
            print("FCGF with KITTI")
        else:
            raise ValueError("Unknown FCGF pretrained model '{}'.".format(model))
        
        checkpoint = torch.hub.load_state_dict_from_url(url, file_name=fname, map_location=self.device)
        self.voxel_size=voxel_size
        self.model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.model = self.model.to(self.device)

    def extract(self, pcd):
        xyz_down, feature = extract_features(
            self.model,
            xyz=np.array(pcd.points),
            voxel_size=self.voxel_size,
            device=self.device,
            skip_check=True)

        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_down)), feature