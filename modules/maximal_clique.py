import numpy as np
import torch
import torch.nn as nn
from igraph import *

import time

def euclidean(a, b):
    return torch.norm(a - b, dim = -1, keepdim = True)

def compatibility(a, b):
    assert(a.shape[-1] == 6)
    assert(b.shape[-1] == 6)
    n1=torch.norm(a[...,:3] - b[...,:3], dim = -1, keepdim = True)
    n2=torch.norm(a[...,3:] - b[...,3:], dim = -1, keepdim = True)
    return torch.abs(n1 - n2)

def Dmatrix(a, type):
    if type == "euclidean":
        return torch.cdist(a, a)
        
    elif type == "compatibility":
        a1 = a[..., :3]
        a2 = a[..., 3:]
        return torch.abs(Dmatrix(a1, "euclidean") - Dmatrix(a2, "euclidean"))
    
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
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

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
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def transform(pts, trans):
    if len(pts.shape) == 3:
        trans_pts = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3],
                                 pts.permute(0, 2, 1)) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = torch.einsum('nm,mk->nk', trans[:3, :3],
                                 pts.T) + trans[:3, 3:4]
        return trans_pts.T

class MaximalCliqueSearcher(nn.Module):
    def __init__(self, inlier_thre, score_thresh, sc2, device = "cuda", show_time = False):
        super().__init__()
        self.device = device
        self.inlier_thre = nn.Parameter(torch.tensor(inlier_thre, dtype=torch.float32)).to(device)
        self.score_thresh = nn.Parameter(torch.tensor(score_thresh, dtype=torch.float32)).to(device)
        self.sc2 = sc2
        self.show_time = show_time

    def construct_graph(self, points):
        '''
        points: B x M x 6
        output: B x M x M
        '''
        dmatrix = Dmatrix(points,"compatibility")
        score = torch.exp(-dmatrix**2 / (2 * self.inlier_thre * self.inlier_thre))
        score[score < self.score_thresh] = 0
        if self.sc2:
            return score * torch.einsum("bmn,bnk->bmk", score, score)
        else:
            return score
    
    def compute_cofficient(self, score):
        sum_fenzi = 0
        sum_fenmu = 1e-10
        cluster_factor = torch.zeros(score.shape[0], dtype = torch.float32, device = self.device)

        score_no_diag = torch.diag(score)
        score_no_diag = torch.diag_embed(score_no_diag)
        score_no_diag = score - score_no_diag

        nonZero = (score_no_diag != 0).to(torch.int32)
        degree = nonZero.sum(dim = 1)

        wij = score_no_diag.unsqueeze(1).repeat(1, score_no_diag.shape[0], 1).permute(0, 2, 1) * score_no_diag.T.unsqueeze(1).repeat(1, score_no_diag.shape[0], 1) # [n, n, n]
        wijk = torch.pow(wij * score_no_diag[None, :, :], 1.0 / 3)
        mask = torch.triu(torch.ones((wijk.shape[1], wijk.shape[2]), dtype = torch.float32, device = self.device), diagonal = 1)
        wijk = wijk * mask[None, :, :]
        wijk = wijk.sum(1).sum(1) # [n]

        invalid = degree <= 1
        degree[invalid] = 0
        f1 = wijk.clone()
        f1[invalid] = 0
        f2 = degree * (degree - 1) * 0.5
        sum_fenzi = f1.sum(0)
        sum_fenmu = f2.sum(0) + 1e-10
        f2[invalid] = 1  # avoid divided by zero
        cluster_factor = f1 / f2

        total_factor = sum_fenzi / sum_fenmu

        return cluster_factor, total_factor
    
    def OTSU_thresh(self, values):
        Quant_num = 100

        score_Hist = torch.zeros(Quant_num, dtype = torch.int32, device = values.device)
        score_sum_Hist = torch.zeros(Quant_num, dtype = torch.float32, device = values.device)

        score_sum = torch.sum(values)
        all_scores, _ = values.sort(0, False)
        max_score_value = all_scores[-1]
        min_score_value = all_scores[0]

        Quant_step = (max_score_value - min_score_value) / Quant_num
        
        ID = (values / Quant_step).to(torch.int32)
        ID[ID >= Quant_num] = Quant_num - 1

        for i in range(ID.shape[0]):
            if ID[i] >= 0:
                score_Hist[ID[i]] = score_Hist[ID[i]] + 1
                score_sum_Hist[ID[i]] = score_sum_Hist[ID[i]] + values[i]

        fmax = -1000
        n1 = 0
        fore_score_sum = 0
        thresh = torch.tensor(0, dtype = torch.float32, device = values.device)
        for i in range(Quant_num):
            Thresh_temp = i * (max_score_value - min_score_value) / Quant_num

            n1 = n1 + score_Hist[i]
            if n1 == 0:
                continue

            n2 = values.shape[0] - n1
            if n2 == 0:
                break

            fore_score_sum = fore_score_sum + score_sum_Hist[i]

            m1 = fore_score_sum / n1
            m2 = (score_sum - fore_score_sum) / n2
            sb = n1 * n2 * pow(m1 - m2, 2)
            if sb > fmax:
                fmax = sb
                thresh = Thresh_temp

        return thresh
    
    def get_graph_matrix(self, score, cluster_factor, cluster_factor_bac, total_factor):
        average_factor = cluster_factor.sum(0) / cluster_factor.shape[0]

        if cluster_factor[0] != 0:
            OTSU = self.OTSU_thresh(cluster_factor.cpu())
        else:
            OTSU = 0

        cluster_threshold = min(OTSU, min(average_factor, total_factor))

        if cluster_threshold > 3 and score.shape[0] > 50:
            f = 10
            tmp = max(OTSU, total_factor)
            if f * tmp > cluster_factor[49]:
                k = (f * tmp - cluster_factor[49]) // (0.05 * tmp) + 1
                f = f - 0.05 * k

            judge_r = (cluster_factor_bac > f * tmp)[:, None]
            judge_c = (cluster_factor_bac <= f * tmp)[None, :]
            judge = judge_r & judge_c
            score[judge] = 0

        score = torch.triu(score, diagonal = 1)

        return score
    
    def node_guided_clique_select(self, macs, score):       
        if len(macs) > 300000:
            macs = macs[0:300000]

        # Gather the weights
        clique_weight = np.zeros(len(macs), dtype = float)
        for ind in range(len(macs)):
            mac = list(macs[ind])
            if len(mac) >= 3:
                score_corr = score[mac, :]
                score_corr = score_corr[:, mac]
                clique_weight[ind] = score_corr.sum(0).sum(0)

        # Find the best clique per-node
        clique_ind_of_node = np.ones(score.shape[0], dtype = int) * -1
        max_clique_weight = np.zeros(score.shape[0], dtype = float)
        max_size = 3
        for ind in range(len(macs)):
            mac = list(macs[ind])
            weight = clique_weight[ind]
            if weight > 0:
                for i in range(len(mac)):
                    if weight > max_clique_weight[mac[i]]:
                        max_clique_weight[mac[i]] = weight
                        clique_ind_of_node[mac[i]] = ind
                        max_size = len(mac) > max_size and len(mac) or max_size

        filtered_clique_ind = list(set(clique_ind_of_node))

        filtered_macs = []
        for ind in filtered_clique_ind:
            filtered_macs.append(macs[ind])

        return filtered_macs
    
    def hypothesis_evaluate(self, points, points_all, macs):       
        points_batch = points_all.repeat([len(macs), 1, 1])
        trans = torch.zeros((len(macs), 4, 4), dtype = torch.float32, device = "cpu")
        for i in range(len(macs)):
            corr = points[0, list(macs[i]), :].to("cpu")
            trans[i, :, :] = rigid_transform_3d(corr[None, :, 0:3], corr[None, :, 3:6], None, 0)[0]
        trans = trans.to(self.device)

        pred_tgt = transform(points_batch[:, :, 0:3], trans)  # [bs,  num_corr, 3]
        L2_dis = torch.norm(pred_tgt - points_all[:, :, 3:6], dim = -1)  # [bs, num_corr]
        MAE_score = torch.div(torch.sub(self.inlier_thre, L2_dis), self.inlier_thre)
        MAE_score = torch.sum(MAE_score * (L2_dis < self.inlier_thre), dim = -1)
        max_batch_score_ind = MAE_score.argmax(dim = -1)
        final_trans = trans[max_batch_score_ind]
        idx = max_batch_score_ind

        return final_trans, idx
    
    def forward(self, points, points_all):
        with torch.no_grad():
            # Construct graph
            t1 = time.perf_counter()
            points = points[None, :, :]
            points_all = points_all[None, :, :]
            score = self.construct_graph(points)[0]
            t2 = time.perf_counter()
            if self.show_time:
                print(f'Graph construction: %.2fms' % ((t2 - t1) * 1000))

            # Reduce graph
            t1 = time.perf_counter()
            cluster_factor, total_factor = self.compute_cofficient(score.clone())
            cluster_factor_bac = cluster_factor.clone()
            cluster_factor, _ = cluster_factor.sort(0, True)
            score = self.get_graph_matrix(score.clone(), cluster_factor, cluster_factor_bac, total_factor)
            t2 = time.perf_counter()
            if self.show_time:
                print(f'Reduce graph: %.2fms' % ((t2 - t1) * 1000))

            # Search cliques
            t1 = time.perf_counter()
            score_np = score.detach().cpu().numpy()
            graph = Graph.Adjacency((score_np > 0).tolist())
            graph.es['weight'] = score_np[score_np.nonzero()]
            graph.vs['label'] = range(0, score_np.shape[0])
            graph.to_undirected()
            macs = graph.maximal_cliques(min = 3)
            t2 = time.perf_counter()
            if self.show_time:
                print(f'Search cliques: %.2fms' % ((t2 - t1) * 1000))

            if len(macs) == 0:
                return torch.eye(4, device = self.device, dtype = torch.float32), []

            # Clique select
            t1 = time.perf_counter()
            filtered_macs = self.node_guided_clique_select(macs, score_np)
            t2 = time.perf_counter()
            if self.show_time:
                print(f'Clique select: %.2fms' % ((t2 - t1) * 1000))

            if len(filtered_macs) == 0:
                return torch.eye(4, device = self.device, dtype = torch.float32), []

            # Hypothesis evaluate
            t1 = time.perf_counter()
            final_trans, idx = self.hypothesis_evaluate(points, points_all, filtered_macs)
            t2 = time.perf_counter()
            if self.show_time:
                print(f'Hypothesis evaluate: %.2fms' % ((t2 - t1) * 1000))

            return final_trans, list(filtered_macs[idx])