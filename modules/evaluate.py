import numpy as np
import math

def InliersCheck(pc0, pc1, matches, pose0, pose1, thre_dis):
    points0 = np.column_stack((pc0, np.ones((len(pc0), 1)))).T
    points1 = np.column_stack((pc1, np.ones((len(pc1), 1)))).T
    points0 = pose0.dot(points0)
    points1 = pose1.dot(points1)

    inliers_num = 0
    for idx in matches:
        dis = (points0[0, idx[0]] - points1[0, idx[1]]) * (points0[0, idx[0]] - points1[0, idx[1]])
        dis = dis + (points0[1, idx[0]] - points1[1, idx[1]]) * (points0[1, idx[0]] - points1[1, idx[1]])
        dis = dis + (points0[2, idx[0]] - points1[2, idx[1]]) * (points0[2, idx[0]] - points1[2, idx[1]])
        dis = math.sqrt(dis)
        if dis < thre_dis:
            inliers_num = inliers_num + 1
    
    return inliers_num

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)