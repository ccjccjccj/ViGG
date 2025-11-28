import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import statistics
from tqdm import tqdm
import datetime
import pickle

from configs import get_cfg_defaults_KITTI

from modules import *

def load_KITTI(image_file, pc_file):
    image = cv2.imread(image_file)

    # Load point cloud
    points = np.fromfile(pc_file, dtype = np.float32).reshape(-1, 4)
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return image, pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", default = True, action = "store_true")
    args = parser.parse_args()

    # Dataset configs to be decided
    default_cfg = get_cfg_defaults_KITTI()
    default_cfg.defrost()
    align_cfg = default_cfg.ALIGN

    # Dataset Parameters
    dataset_cfg = default_cfg.DATASET

    # Registration
    register = ViGG(align_cfg, "KITTI")
    err_R, err_t, inlier_num, vis_recall, eps = [], [], [], [], []

    with open(os.path.join("data/kitti_test.pkl"), 'rb') as f:
        metadata_list = pickle.load(f)

    for meta in tqdm(metadata_list):
        rgb0_path = meta["rgb0_path"]
        rgb1_path = meta["rgb1_path"]
        pc0_path = meta["pc0_path"]
        pc1_path = meta["pc1_path"]

        # Read file with opencv and open3d
        img0, pc0 = load_KITTI(dataset_cfg.kitti_root + '/' + rgb0_path, dataset_cfg.kitti_root + '/' + pc0_path)
        img1, pc1 = load_KITTI(dataset_cfg.kitti_root + '/'  + rgb1_path, dataset_cfg.kitti_root + '/'  + pc1_path)
        print(rgb0_path)
        print(rgb1_path)

        camera_matrix = meta["camera_matrix"]
        vel_to_cam = meta["vel_to_cam"]
        transform = np.linalg.inv(meta["transform"])

        # Remove the points behind the camera
        idx_img0, idx_img1 = [], []
        points0 = np.array(pc0.points)
        points0 = np.concatenate((points0, np.ones((points0.shape[0], 1))), axis = 1)
        points_tmp = (vel_to_cam @ points0.T).T
        for n in range(points_tmp.shape[0]):
            if points_tmp[n, 2] > 0:
                idx_img0.append(n)
        points0 = points0[idx_img0, :]
        pc0_clip = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points0[:, 0:3]))
        points1 = np.array(pc1.points)
        points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis = 1)
        points_tmp = (vel_to_cam @ points1.T).T
        for n in range(points_tmp.shape[0]):
            if points_tmp[n, 2] > 0:
                idx_img1.append(n)
        points1 = points1[idx_img1, :]
        pc1_clip = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1[:, 0:3]))

        # Remove the points out of the image
        points0 = np.array(pc0_clip.points)
        points0 = np.concatenate((points0, np.ones((points0.shape[0], 1))), axis = 1)
        points1 = np.array(pc1_clip.points)
        points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis = 1)
        uv0 = camera_matrix @ points0.T
        uv1 = camera_matrix @ points1.T
        uv0 = (uv0 / uv0[2, :]).T
        uv1 = (uv1 / uv1[2, :]).T
        idx_img0, idx_img1 = [], []
        for n in range(uv0.shape[0]):
            if uv0[n, 0] >=0 and uv0[n, 0] < 1226:
                if uv0[n, 1] >=0 and uv0[n, 1] < 370:
                    idx_img0.append(n)
        for n in range(uv1.shape[0]):
            if uv1[n, 0] >=0 and uv1[n, 0] < 1226:
                if uv1[n, 1] >=0 and uv1[n, 1] < 370:
                    idx_img1.append(n)
        points0 = points0[idx_img0, :]
        points1 = points1[idx_img1, :]
        pc0_clip = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points0[:, 0:3]))
        pc1_clip = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1[:, 0:3]))

        # Align
        result = register.align_KITTI(img0, img1, pc0, pc1, pc0_clip, pc1_clip, camera_matrix)
        res = result["trans"]
        pc0_corr = result["pc0_corr"]
        pc1_corr = result["pc1_corr"]
        visual_recall = result["visual_recall"]
        epsilon = result["epsilon"]

        # Check the inlier
        matches = []
        for i in range(pc0_corr.shape[0]):
            matches.append([i, i])
        inlier_num_filter = [InliersCheck(pc0_corr, pc1_corr, matches, transform, np.eye(4), thre_dis) for thre_dis in [1.2, 0.6, 0.3]]
        print(f"Inlier:     matches num:{len(matches)}  1.2m inliers:{inlier_num_filter[0]}  0.6m inliers:{inlier_num_filter[1]}  0.3m inliers:{inlier_num_filter[2]}")
        inlier_num.append([len(matches), inlier_num_filter[0], inlier_num_filter[1], inlier_num_filter[2]])

        # Coumpute error
        RRE = compute_relative_rotation_error(transform[0:3, 0:3], res[0:3, 0:3])
        RTE = compute_relative_translation_error(transform[0:3, 3], res[0:3, 3])
        print(f"RRE: {RRE}, RTE: {RTE}")
        err_R.append(RRE)
        err_t.append(RTE)

        vis_recall.append(visual_recall)
        eps.append(epsilon)

    # Evaluate
    racc = np.array([len([x for x in err_R if x <= 0.25]), len([x for x in err_R if x <= 0.5]), len([x for x in err_R if x <= 1])], np.float64) / len(err_R) * 100
    tacc = np.array([len([x for x in err_t if x <= 0.075]), len([x for x in err_t if x <= 0.15]), len([x for x in err_t if x <= 0.3])], np.float64) / len(err_t) * 100
    inlier_np = np.array(inlier_num, np.float64)
    for i in range(inlier_np.shape[0]):
        #  Make 0 match num has 0% inlier ratio instead of inf
        if inlier_np[i, 0] == 0:
            inlier_np[i, 0] = 1
    inlier_ratio_mean = np.array([np.mean(inlier_np[:, 1] / inlier_np[:, 0]), np.mean(inlier_np[:, 2] / inlier_np[:, 0]), np.mean(inlier_np[:, 3] / inlier_np[:, 0])])
    inlier_ratio_median = np.array([np.median(inlier_np[:, 1] / inlier_np[:, 0]), np.median(inlier_np[:, 2] / inlier_np[:, 0]), np.median(inlier_np[:, 3] / inlier_np[:, 0])])
    
    # Recall
    recall = 0
    for i in range(len(err_R)):
        if err_R[i] < 5 and err_t[i] < 0.6:
            recall += 1

    # Print
    print(f"inlier_ratio_Mean:{inlier_ratio_mean}")
    print(f"inlier_ratio_Med:{inlier_ratio_median}")
    print(f"racc:{racc}")
    print(f"tacc:{tacc}")
    print(f"err_R_Med:{statistics.median(err_R)}")
    print(f"err_t_Med:{statistics.median(err_t)}")
    print(f"recall:{recall / len(err_R) * 100}")

    # Save
    if not os.path.exists("result"):
        os.mkdir("result")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(f"result/KITTI_{timestamp}")
    np.save(f"result/KITTI_{timestamp}/err_R.npy", np.array(err_R))
    np.save(f"result/KITTI_{timestamp}/err_t.npy", np.array(err_t))
    np.save(f"result/KITTI_{timestamp}/inlier_num.npy", np.array(inlier_num))
    np.save(f"result/KITTI_{timestamp}/visual_recall.npy", np.array(vis_recall))
    np.save(f"result/KITTI_{timestamp}/epsilon.npy", np.array(eps))
    with open(f"result/KITTI_{timestamp}/cfg.pkl", "wb") as file:
        pickle.dump(align_cfg, file)