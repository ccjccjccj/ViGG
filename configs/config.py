import os

from yacs.config import CfgNode as CN

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
REPO_PATH = os.path.dirname(PROJECT_PATH)

_C = CN()

_C.DATASET = CN()
_C.ALIGN = CN()

# Dataset Parameters
_C.DATASET.project_root = PROJECT_PATH
_C.DATASET.rgbd_3d_root = os.path.join("--Need-To-Set--")
_C.DATASET.scannet_root = os.path.join("--Need-To-Set--")
_C.DATASET.name = "3DMatch"  # 3DMatch or ScanNet
_C.DATASET.view_spacing = 60  # Frame spacing
# -- loader params
_C.DATASET.num_workers = 4
_C.DATASET.overfit = False
_C.DATASET.batch_size = 4

# ViGG Parameters
_C.ALIGN.voxel_size = 0.025  # Downsample voxel size
_C.ALIGN.SampleK = 5000  # Reduce the upper bound of GPU memory cost (2GB for 5000), may causes accuracy loss when SampleK is far less than the points num
_C.ALIGN.pointcloud_feature = "FCGF"  # FPFH or FCGF
_C.ALIGN.image_feature = "LightGlue"  # SIFT or LightGlue
_C.ALIGN.topK = 150  # Sample topK visual correspondences for fast maximal clique registration, increase it will result in higher accuracy but lower speed
_C.ALIGN.inlier_thre = 0.1  # Inlier threshold
_C.ALIGN.score_thre = 0.99  # Score threshold 0.9 for SIFT and 0.99 for LightGlue

# Parameters for Ablation Expierments (See the ablations in our paper)
_C.ALIGN.add_noise = 0  # Add gaussian noise to visual correspondences, only used for ablation
_C.ALIGN.conf_bar = 10  # Quantile
_C.ALIGN.iters = 3  # Iterations of point-wise local match
_C.ALIGN.algorithm = "SVD"  # RANSAC or SVD

def get_cfg_defaults():
    return _C.clone()