import torch
import numpy as np

def weighted_svd(pc0, pc1, dists):
    if pc0.shape[0] == 0:
        return torch.eye(4, device = pc0.device, dtype = torch.float32)

    corr_P0 = pc0[None, :, :]
    corr_P1 = pc1[None, :, :]
    dists = dists[None, :]

    # Transform euclidean distance of points on a sphere to cosine similarity
    cosine = 1 - 0.5 * dists
    Rt = rigid_transform_3d(corr_P0, corr_P1, cosine)[0, :, :]
    
    return Rt

def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T
    
def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans
    
def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def post_refinement(initial_trans, src_keypts, tgt_keypts, inlier_thre):
    """
    Perform post refinement using the initial transformation matrix.
    Input
        - initial_trans: [4, 4] 
        - src_keypts:    [num_corr, 3]    
        - tgt_keypts:    [num_corr, 3]
        - corres:        [num_corr, 2]
    Output:    
        - final_trans:   [4, 4]
    """
    if src_keypts.shape[0] == 0:
        return initial_trans

    src_keypts = src_keypts[None, :, :]
    tgt_keypts = tgt_keypts[None, :, :]
    initial_trans = initial_trans[None, :, :]

    inlier_threshold_list = [inlier_thre] * 20

    previous_inlier_num = 0
    for inlier_threshold in inlier_threshold_list:
        warped_src_keypts = transform(src_keypts, initial_trans)
        L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
        inlier_num = torch.sum(pred_inlier)
        if abs(int(inlier_num - previous_inlier_num)) < 1:
            break
        else:
            previous_inlier_num = inlier_num
        initial_trans = rigid_transform_3d(
            A=src_keypts[:, pred_inlier, :],
            B=tgt_keypts[:, pred_inlier, :],
            weights=1/(1 + (L2_dis/inlier_threshold)**2)[:, pred_inlier],
        )

    return initial_trans[0, :, :]