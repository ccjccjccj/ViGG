import random
import numpy as np
import torch
import open3d as o3d

from .utils import FeatureMatch, CalEpsilon, Pixel2Point, Find3DMapping, PointwiseSearchZone, PointwiseLocalMatch
from .image_extractor import ImageExtractor
from .pointcloud_extractor import PointCloudExtractor
from .maximal_clique import MaximalCliqueSearcher
from .alignment import weighted_svd, post_refinement

class ViGG:
    def __init__(self, align_cfg, model = "3DMatch", device = "cuda"):
        print("ViGG Settings")
        print(f"Voxel Size:{align_cfg.voxel_size}")
        print(f"SampleK:{align_cfg.SampleK}")
        print(f"PointCloud Feature:{align_cfg.pointcloud_feature}")
        print(f"Image Feature:{align_cfg.image_feature}")
        print(f"TopK:{align_cfg.topK}")
        print(f"Inlier Thre:{align_cfg.inlier_thre}")
        print(f"Score Thre:{align_cfg.score_thre}")
        print("Ablations Settings")
        print(f"Noise:{align_cfg.add_noise}")
        print(f"Confidence Bar:{align_cfg.conf_bar}")
        print(f"Iters:{align_cfg.iters}")
        print(f"Algorithm:{align_cfg.algorithm}")
        print("================================")
        
        self.cfg = align_cfg
        self.add_image_matches = True
        self.extractor_img = ImageExtractor(device = device)
        self.extractor_pc = PointCloudExtractor(self.cfg.voxel_size, model, device = device)
        self.maximal_clique = MaximalCliqueSearcher(self.cfg.inlier_thre, self.cfg.score_thre, True, device = device)
        self.device = device
    
    def align(self, img0, img1, dep0, dep1, intrinsics):
        assert img0.shape == img1.shape

        # Generate PointClouds
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(width = img0.shape[0], height = img0.shape[1], fx = intrinsics[0,0], fy = intrinsics[1,1], cx = intrinsics[0,2], cy = intrinsics[1,2])
        # Invalid depth with 65535 and 0 will be removed
        pc0 = o3d.geometry.PointCloud.create_from_depth_image(dep0, camera_intrinsic, depth_scale = 1000, depth_trunc = 65.535)
        pc1 = o3d.geometry.PointCloud.create_from_depth_image(dep1, camera_intrinsic, depth_scale = 1000, depth_trunc = 65.535)

        # Extract point cloud features
        if self.cfg.pointcloud_feature == "FPFH":
            pc0_down, pc0_des = self.extractor_pc.FPFH(pc0)
            pc1_down, pc1_des = self.extractor_pc.FPFH(pc1)
        elif self.cfg.pointcloud_feature == "FCGF":
            pc0_down, pc0_des = self.extractor_pc.FCGF(pc0)
            pc1_down, pc1_des = self.extractor_pc.FCGF(pc1)
        elif self.cfg.pointcloud_feature == None:
            pc0_down = pc0.voxel_down_sample(self.cfg.voxel_size)
            pc1_down = pc1.voxel_down_sample(self.cfg.voxel_size)
            pc0_des = []
            pc1_des = []
        else:
            raise ValueError("Unknown pointcloud feature '{}'.".format(self.cfg.pointcloud_feature))
        
        # Sample points for lower computational cost
        pc0_pts = np.asarray(pc0_down.points).astype(np.float32)
        pc1_pts = np.asarray(pc1_down.points).astype(np.float32)
        if pc0_pts.shape[0] > self.cfg.SampleK:
            idx = random.sample(range(pc0_pts.shape[0]), k = self.cfg.SampleK)
            pc0_pts = pc0_pts[idx, :]
            if len(pc0_des) > 0:
                pc0_des = pc0_des[idx, :]

        pc0_pts = torch.tensor(pc0_pts, device = self.device)
        pc1_pts = torch.tensor(pc1_pts, device = self.device)
        pc0_des = torch.tensor(pc0_des, device = self.device)
        pc1_des = torch.tensor(pc1_des, device = self.device)

        # Extract image features
        if self.cfg.image_feature == "SIFT":
            kp0, des0 = self.extractor_img.SIFT(img0)
            kp1, des1 = self.extractor_img.SIFT(img1)
        elif self.cfg.image_feature == "LightGlue":
            kp0, kp1, dists_vis = self.extractor_img.LightGlue(img0, img1)
        elif self.cfg.image_feature == None:
            kp0 = torch.tensor([], device = self.device)
            kp1 = torch.tensor([], device = self.device)
            des0 = torch.tensor([], device = self.device)
            des1 = torch.tensor([], device = self.device)
        else:
            raise ValueError("Unknown image feature '{}'.".format(self.cfg.image_feature))
        
        # Match image features
        if not self.cfg.image_feature == "LightGlue":
            matches_img, dists_vis = FeatureMatch(des0, des1, 0.8)
            kp0 = kp0[matches_img[:, 0]]
            kp1 = kp1[matches_img[:, 1]]

        # Image to point cloud
        dep0 = torch.tensor(np.asarray(dep0, np.float32), device = self.device)
        dep1 = torch.tensor(np.asarray(dep1, np.float32), device = self.device)
        intrinsics = torch.tensor(intrinsics, device = self.device, dtype = torch.float32)
        pc0_vis, pc1_vis, mask = Pixel2Point(kp0, dep0, kp1, dep1, intrinsics)
        pc0_vis = pc0_vis[mask]
        pc1_vis = pc1_vis[mask]
        dists_vis = dists_vis[mask]

        # Reduce match num
        if pc0_vis.shape[0] > self.cfg.topK:
            dists_vis, indices = torch.sort(dists_vis)
            dists_vis = dists_vis[0:self.cfg.topK]
            indices = indices[0:self.cfg.topK]
            pc0_vis = pc0_vis[indices, :]
            pc1_vis = pc1_vis[indices, :]

        # Add noise for ablation experiments
        if self.cfg.add_noise > 0:
            noise = torch.normal(0, self.cfg.add_noise, size = pc0_vis.shape, device = self.device)
            pc0_vis = pc0_vis + noise
            noise = torch.normal(0, self.cfg.add_noise, size = pc1_vis.shape, device = self.device)
            pc1_vis = pc1_vis + noise

        # Geometric matches
        matches_pc, _ = FeatureMatch(pc0_des, pc1_des)
        pc0_geo = pc0_pts[matches_pc[:, 0]]
        pc1_geo = pc1_pts[matches_pc[:, 1]]

        # Maximal clique registration & Get epsilon
        if pc0_vis.shape[0] < 3:
            trans_tmp = torch.eye(4, device = self.device, dtype = torch.float32)
            mask = torch.zeros(pc0_vis.shape[0], device = self.device, dtype = torch.bool)
            epsilon = 10000
        else:
            corr_pts = torch.cat([pc0_vis, pc1_vis], axis = 1)
            pc0_corr_all = torch.cat([pc0_geo, pc0_vis], axis = 0)
            pc1_corr_all = torch.cat([pc1_geo, pc1_vis], axis = 0)
            corr_pts_all = torch.cat([pc0_corr_all, pc1_corr_all], axis = 1)
            trans_tmp, _ = self.maximal_clique(corr_pts, corr_pts_all)
            mask, epsilon = CalEpsilon(pc0_vis, pc1_vis, trans_tmp, self.cfg.inlier_thre, self.cfg.conf_bar)

        pc0_vis_in = pc0_vis[mask]
        pc1_vis_in = pc1_vis[mask]
        dists_vis_in = dists_vis[mask]

        if self.cfg.pointcloud_feature == None:
            # Post refinement
            if epsilon >= 10000:
                final_trans = post_refinement(trans_tmp, pc0_vis, pc1_vis, self.cfg.inlier_thre)
            else:
                final_trans = post_refinement(trans_tmp, pc0_vis, pc1_vis, pow(epsilon, 0.5) / 2)

            # Collect results
            result = {}
            result["trans"] = final_trans.cpu().numpy()
            result["pc0_corr"] = pc0_vis.cpu().numpy()
            result["pc1_corr"] = pc1_vis.cpu().numpy()
            result["visual_recall"] = 0
            result["epsilon"] = epsilon

            return result

        for i in range(self.cfg.iters):
            # Extract correspondences
            if epsilon >= 10000:
                matches_pc, dists_geo = FeatureMatch(pc0_des, pc1_des)
                visual_recall = 0
                i = self.cfg.iters
            else:
                zone = PointwiseSearchZone(pc0_pts, pc1_pts, trans_tmp, epsilon)
                matches_pc, dists_geo = PointwiseLocalMatch(pc0_des, pc1_des, zone)
                visual_recall = 1

            pc0_geo = pc0_pts[matches_pc[:, 0]]
            pc1_geo = pc1_pts[matches_pc[:, 1]]
            
            # Add image matches
            pc0_corr = pc0_geo  # (num_points, 3)
            pc1_corr = pc1_geo
            dists_corr = dists_geo
            if self.add_image_matches:
                pc0_corr = torch.cat([pc0_corr, pc0_vis_in], axis = 0)
                pc1_corr = torch.cat([pc1_corr, pc1_vis_in], axis = 0)
                dists_corr = torch.cat([dists_corr, dists_vis_in], axis = 0)

            # Estimate tranformation
            if self.cfg.algorithm == "SVD":
                trans_tmp = weighted_svd(pc0_corr, pc1_corr, dists_corr)
            elif self.cfg.algorithm == "RANSAC":
                pc0_o3d = pc0_corr.cpu().numpy()
                pc1_o3d = pc1_corr.cpu().numpy()
                corres = []
                for i in range(pc0_o3d.shape[0]):
                    corres.append([i, i])
                pc0_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc0_o3d))
                pc1_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1_o3d))
                trans_tmp = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pc0_o3d, pc1_o3d, o3d.utility.Vector2iVector(corres), self.cfg.inlier_thre, ransac_n = 3).transformation
                trans_tmp = torch.tensor(trans_tmp, device = self.device, dtype = torch.float32)
            else:
                raise ValueError("Unknown algorithm '{}'.".format(self.cfg.algorithm))
            
            # Update epsilon
            if i < self.cfg.iters - 1:
                mask, epsilon_tmp = CalEpsilon(pc0_vis, pc1_vis, trans_tmp, self.cfg.inlier_thre, self.cfg.conf_bar)
                if epsilon_tmp < epsilon:
                    epsilon = epsilon_tmp
                    pc0_vis_in = pc0_vis[mask]
                    pc1_vis_in = pc1_vis[mask]
                    dists_vis_in = dists_vis[mask]
            else:
                break
        
        # Post refinement
        if epsilon >= 10000:
            final_trans = post_refinement(trans_tmp, pc0_corr, pc1_corr, self.cfg.inlier_thre)
        else:
            final_trans = post_refinement(trans_tmp, pc0_corr, pc1_corr, pow(epsilon, 0.5) / 2)

        # Collect results
        result = {}
        result["trans"] = final_trans.cpu().numpy()
        result["pc0_corr"] = pc0_corr.cpu().numpy()
        result["pc1_corr"] = pc1_corr.cpu().numpy()
        result["visual_recall"] = visual_recall
        result["epsilon"] = epsilon

        return result
    
    def align_KITTI(self, img0, img1, pc0, pc1, pc0_clip, pc1_clip, camera_matrix):
        assert img0.shape == img1.shape

        # Extract point cloud features
        if self.cfg.pointcloud_feature == "FPFH":
            pc0_down, pc0_des = self.extractor_pc.FPFH(pc0)
            pc1_down, pc1_des = self.extractor_pc.FPFH(pc1)
        elif self.cfg.pointcloud_feature == "FCGF":
            pc0_down, pc0_des = self.extractor_pc.FCGF(pc0)
            pc1_down, pc1_des = self.extractor_pc.FCGF(pc1)
        elif self.cfg.pointcloud_feature == None:
            pc0_down = pc0.voxel_down_sample(self.cfg.voxel_size)
            pc1_down = pc1.voxel_down_sample(self.cfg.voxel_size)
            pc0_des = []
            pc1_des = []
        else:
            raise ValueError("Unknown pointcloud feature '{}'.".format(self.cfg.pointcloud_feature))
        
        # Sample points for lower computational cost
        pc0_pts = np.asarray(pc0_down.points).astype(np.float32)
        pc1_pts = np.asarray(pc1_down.points).astype(np.float32)
        pc0_clip = np.asarray(pc0_clip.points).astype(np.float32)
        pc1_clip = np.asarray(pc1_clip.points).astype(np.float32)
        if pc0_pts.shape[0] > self.cfg.SampleK:
            idx = random.sample(range(pc0_pts.shape[0]), k = self.cfg.SampleK)
            pc0_pts = pc0_pts[idx, :]
            if len(pc0_des) > 0:
                pc0_des = pc0_des[idx, :]

        pc0_pts = torch.tensor(pc0_pts, device = self.device)
        pc1_pts = torch.tensor(pc1_pts, device = self.device)
        pc0_des = torch.tensor(pc0_des, device = self.device)
        pc1_des = torch.tensor(pc1_des, device = self.device)
        pc0_clip = torch.tensor(pc0_clip, device = self.device)
        pc1_clip = torch.tensor(pc1_clip, device = self.device)

        # Extract image features
        if self.cfg.image_feature == "SIFT":
            kp0, des0 = self.extractor_img.SIFT(img0)
            kp1, des1 = self.extractor_img.SIFT(img1)
        elif self.cfg.image_feature == "LightGlue":
            kp0, kp1, dists_vis = self.extractor_img.LightGlue(img0, img1)
        elif self.cfg.image_feature == None:
            kp0 = torch.tensor([], device = self.device)
            kp1 = torch.tensor([], device = self.device)
            des0 = torch.tensor([], device = self.device)
            des1 = torch.tensor([], device = self.device)
        else:
            raise ValueError("Unknown image feature '{}'.".format(self.cfg.image_feature))
        
        # Match image features
        if not self.cfg.image_feature == "LightGlue":
            matches_img, dists_vis = FeatureMatch(des0, des1, 0.8)
            kp0 = kp0[matches_img[:, 0]]
            kp1 = kp1[matches_img[:, 1]]

        # Image to point cloud
        if self.cfg.image_feature == None:
            pc0_vis = torch.tensor([], device = self.device)
            pc1_vis = torch.tensor([], device = self.device)
        else:
            # Image correspondences to point cloud correspondences
            camera_matrix = torch.tensor(camera_matrix, device = self.device, dtype = torch.float32)
            mp0, mask0 = Find3DMapping(kp0, pc0_clip, camera_matrix, 5)
            mp1, mask1 = Find3DMapping(kp1, pc1_clip, camera_matrix, 5)
            mask = mask0 & mask1
            pc0_vis = mp0[mask, :]
            pc1_vis = mp1[mask, :]
            dists_vis = dists_vis[mask]

        # Reduce match num
        if pc0_vis.shape[0] > self.cfg.topK:
            dists_vis, indices = torch.sort(dists_vis)
            dists_vis = dists_vis[0:self.cfg.topK]
            indices = indices[0:self.cfg.topK]
            pc0_vis = pc0_vis[indices, :]
            pc1_vis = pc1_vis[indices, :]

        # Add noise for ablation experiments
        if self.cfg.add_noise > 0:
            noise = torch.normal(0, self.cfg.add_noise, size = pc0_vis.shape, device = self.device)
            pc0_vis = pc0_vis + noise
            noise = torch.normal(0, self.cfg.add_noise, size = pc1_vis.shape, device = self.device)
            pc1_vis = pc1_vis + noise

        # Geometric matches
        matches_pc, _ = FeatureMatch(pc0_des, pc1_des)
        pc0_geo = pc0_pts[matches_pc[:, 0]]
        pc1_geo = pc1_pts[matches_pc[:, 1]]

        # Maximal clique registration & Get epsilon
        if pc0_vis.shape[0] < 3:
            trans_tmp = torch.eye(4, device = self.device, dtype = torch.float32)
            mask = torch.zeros(pc0_vis.shape[0], device = self.device, dtype = torch.bool)
            epsilon = 10000
        else:
            corr_pts = torch.cat([pc0_vis, pc1_vis], axis = 1)
            pc0_corr_all = torch.cat([pc0_geo, pc0_vis], axis = 0)
            pc1_corr_all = torch.cat([pc1_geo, pc1_vis], axis = 0)
            corr_pts_all = torch.cat([pc0_corr_all, pc1_corr_all], axis = 1)
            trans_tmp, _ = self.maximal_clique(corr_pts, corr_pts_all)
            mask, epsilon = CalEpsilon(pc0_vis, pc1_vis, trans_tmp, self.cfg.inlier_thre, self.cfg.conf_bar)

        pc0_vis_in = pc0_vis[mask]
        pc1_vis_in = pc1_vis[mask]
        dists_vis_in = dists_vis[mask]

        if self.cfg.pointcloud_feature == None:
            # Post refinement
            if epsilon >= 10000:
                final_trans = post_refinement(trans_tmp, pc0_vis, pc1_vis, self.cfg.inlier_thre)
            else:
                final_trans = post_refinement(trans_tmp, pc0_vis, pc1_vis, pow(epsilon, 0.5) / 2)

            # Collect results
            result = {}
            result["trans"] = final_trans.cpu().numpy()
            result["pc0_corr"] = pc0_vis.cpu().numpy()
            result["pc1_corr"] = pc1_vis.cpu().numpy()
            result["visual_recall"] = 0
            result["epsilon"] = epsilon

            return result

        for i in range(self.cfg.iters):
            # Extract correspondences
            if epsilon >= 10000:
                matches_pc, dists_geo = FeatureMatch(pc0_des, pc1_des)
                visual_recall = 0
                i = self.cfg.iters
            else:
                zone = PointwiseSearchZone(pc0_pts, pc1_pts, trans_tmp, epsilon)
                matches_pc, dists_geo = PointwiseLocalMatch(pc0_des, pc1_des, zone)
                visual_recall = 1

            pc0_geo = pc0_pts[matches_pc[:, 0]]
            pc1_geo = pc1_pts[matches_pc[:, 1]]

            # Add image matches
            pc0_corr = pc0_geo  # (num_points, 3)
            pc1_corr = pc1_geo
            dists_corr = dists_geo
            if self.add_image_matches:
                pc0_corr = torch.cat([pc0_corr, pc0_vis_in], axis = 0)
                pc1_corr = torch.cat([pc1_corr, pc1_vis_in], axis = 0)
                dists_corr = torch.cat([dists_corr, dists_vis_in], axis = 0)

            # Estimate tranformation
            if self.cfg.algorithm == "SVD":
                trans_tmp = weighted_svd(pc0_corr, pc1_corr, dists_corr)
            elif self.cfg.algorithm == "RANSAC":
                pc0_o3d = pc0_corr.cpu().numpy()
                pc1_o3d = pc1_corr.cpu().numpy()
                corres = []
                for i in range(pc0_o3d.shape[0]):
                    corres.append([i, i])
                pc0_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc0_o3d))
                pc1_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1_o3d))
                trans_tmp = o3d.pipelines.registration.registration_ransac_based_on_correspondence(pc0_o3d, pc1_o3d, o3d.utility.Vector2iVector(corres), self.cfg.inlier_thre, ransac_n = 3).transformation
                trans_tmp = torch.tensor(trans_tmp, device = self.device, dtype = torch.float32)
            else:
                raise ValueError("Unknown algorithm '{}'.".format(self.cfg.algorithm))

            # Update epsilon
            if i < self.cfg.iters - 1:
                mask, epsilon_tmp = CalEpsilon(pc0_vis, pc1_vis, trans_tmp, self.cfg.inlier_thre, self.cfg.conf_bar)
                if epsilon_tmp < epsilon:
                    epsilon = epsilon_tmp
                    pc0_vis_in = pc0_vis[mask]
                    pc1_vis_in = pc1_vis[mask]
                    dists_vis_in = dists_vis[mask]
            else:
                break
        
        # Post refinement
        if epsilon >= 10000:
            final_trans = post_refinement(trans_tmp, pc0_corr, pc1_corr, self.cfg.inlier_thre)
        else:
            final_trans = post_refinement(trans_tmp, pc0_corr, pc1_corr, pow(epsilon, 0.5) / 2)

        # Collect results
        result = {}
        result["trans"] = final_trans.cpu().numpy()
        result["pc0_corr"] = pc0_corr.cpu().numpy()
        result["pc1_corr"] = pc1_corr.cpu().numpy()
        result["visual_recall"] = visual_recall
        result["epsilon"] = epsilon

        return result