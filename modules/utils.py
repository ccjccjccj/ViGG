import numpy as np
import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import ball_query

def FeatureMatch(des0, des1, lowe_ratio = 0):
    if des0.shape[0] == 0 or des1.shape[0] == 0:
        dists = torch.tensor([], device = des0.device, dtype = torch.int64)
        matches = torch.tensor([], device = des0.device, dtype = torch.int64)[:, None]
        return torch.cat([matches, matches], dim = 1), dists
    
    # We use direct nearest neighbor method to find the correspondences
    P0 = des0[None, :, :]
    P1 = des1[None, :, :]

    # Normalize points for cosine distance
    P0 = P0 / P0.norm(dim = 2, keepdim = True).clamp(min = 1e-9)
    P1 = P1 / P1.norm(dim = 2, keepdim = True).clamp(min = 1e-9)
    
    if lowe_ratio > 0:
        dists, matches, _ = knn_points(P0, P1, K = 2)
        idx = []
        for i in range(matches.shape[1]):
            if dists[0, i, 0] < lowe_ratio * dists[0, i, 1]:
                idx.append(i)
        matches = matches[0, idx, 0]
        dists = dists[0, idx, 0]
        matches = torch.cat([torch.tensor(idx, device = matches.device, dtype = torch.int64)[:, None], matches[:, None]], dim = 1)
    else:
        dists, matches, _ = knn_points(P0, P1, K = 1)
        matches = matches[0, :, 0]
        dists = dists[0, :, 0]
        matches = torch.cat([torch.arange(0, matches.shape[0], 1, device = matches.device, dtype = torch.int64)[:, None], matches[:, None]], dim = 1)
    
    return matches, dists

def CalEpsilon(pc0_vis, pc1_vis, trans, inlier_thre, conf_bar, K = 1):
    # Cal errors of inliers
    dis = 0
    mp0 = torch.cat([pc0_vis, torch.ones((pc0_vis.shape[0], 1), device = pc0_vis.device)], axis = 1).T
    mp1 = torch.cat([pc1_vis, torch.ones((pc1_vis.shape[0], 1), device = pc0_vis.device)], axis = 1).T
    mp0 = trans @ mp0
    
    dx = mp0[0, :] - mp1[0, :]
    dy = mp0[1, :] - mp1[1, :]
    dz = mp0[2, :] - mp1[2, :]
    d = dx * dx + dy * dy + dz * dz
    mask = d <= (K * inlier_thre) * (K * inlier_thre)

    # Cal epsilon
    if d[mask].shape[0] >= 3:
        dis = d[mask].sum()
        sigma = dis / d[mask].shape[0] / 3
        epsilon = sigma * conf_bar
        return mask, epsilon.cpu().item()  # returned epsilon is actually the squared value
    else:
        epsilon = 10000
        return mask, epsilon

def PointCloudToUV(pc, intrinsics):
    points = np.asarray(pc.points)  # (num_points, 3)
    u = ((points[:, 0] / points[:, 2]) * intrinsics[0, 0] + intrinsics[0, 2])
    v = ((points[:, 1] / points[:, 2]) * intrinsics[1, 1] + intrinsics[1, 2])
    return u, v

def Find3DMapping(kp, pc, camera_matrix, dis_thre):
    # Point cloud to image
    points = torch.cat([pc, torch.ones((pc.shape[0], 1), device = pc.device, dtype = torch.float32)], axis = 1)
    uv = (camera_matrix @ points.T)
    uv = (uv / uv[2, :]).T[:, 0:2]

    # Get 3D mapping indexs of 2D correspondences
    P0 = kp[None, :, :]
    P1 = uv[None, :, :]
    dists, matches, _ = knn_points(P0, P1, K = 1)
    dists = dists[0, :, 0]**0.5
    matches = matches[0, :, 0]
    mask = dists <= dis_thre

    return points[matches, 0:3], mask

def Pixel2Point(kp0, dep0, kp1, dep1, intrinsics):
    depth0 = dep0.T / 1000
    mp0 = kp0.to(torch.int32)
    z0 = depth0[mp0[:, 0], mp0[:, 1]]
    depth1 = dep1.T / 1000
    mp1 = kp1.to(torch.int32)
    z1 = depth1[mp1[:, 0], mp1[:, 1]]

    mask0 = (z0 < 65.535) & (z0 > 0)
    mask1 = (z1 < 65.535) & (z1 > 0)
    mask = mask0 & mask1

    # UV to XYZ
    x0 = (mp0[:, 0] - intrinsics[0, 2]) * z0 / intrinsics[0, 0]
    x1 = (mp1[:, 0] - intrinsics[0, 2]) * z1 / intrinsics[0, 0]
    y0 = (mp0[:, 1] - intrinsics[1, 2]) * z0 / intrinsics[1, 1]
    y1 = (mp1[:, 1] - intrinsics[1, 2]) * z1 / intrinsics[1, 1]

    points0 = torch.cat([x0[:, None], y0[:, None], z0[:, None]], axis = 1)
    points1 = torch.cat([x1[:, None], y1[:, None], z1[:, None]], axis = 1)

    return points0, points1, mask

def PointwiseSearchZone(src_points, des_points, trans, epsilon):
    # Transform
    src_trans = (trans @ torch.cat((src_points, torch.ones((src_points.shape[0], 1), device = src_points.device)), dim = 1).T).T[:, 0:3]
    _, matches, _ = ball_query(src_trans[None, :, :], des_points[None, :, :], radius = pow(epsilon, 0.5), K = 500)

    return matches[0]

def PointwiseLocalMatch(feats0, feats1, zone):
    # Normalize points for cosine distance
    P0 = feats0 / feats0.norm(dim = 1, keepdim = True).clamp(min = 1e-9)
    P1 = feats1 / feats1.norm(dim = 1, keepdim = True).clamp(min = 1e-9)

    # similarity zone_src and p_ref
    map1 = torch.cat((P1, 1e5 * torch.ones((1, feats1.shape[1]), device = P1.device)), dim = 0)
    map1 = map1[zone, :]
    dist = P0.unsqueeze(1) - map1
    dists = (dist * dist).sum(2)
    value, idx = torch.min(dists, 1)

    valid = (value < 100).to(torch.int32).nonzero().squeeze(1)
    corr = zone[valid, idx[valid]]
    corr_d = value[valid]
    corr_idxs = torch.cat([valid[:, None], corr[:, None]], dim = 1)

    return corr_idxs, corr_d