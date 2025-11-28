import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import statistics
from tqdm import tqdm
import datetime
import pickle

from datasets import build_loader
from configs import get_cfg_defaults

from modules import *

def load_3DMatch(path):
    image_file = path
    depth_file = os.path.dirname(path) + '/' + os.path.basename(path).replace("color", "depth")
    pose_file =  os.path.dirname(path) + '/' + os.path.basename(path).replace("color.png", "pose.txt")
    intrinsics_file = os.path.dirname(os.path.dirname(path)) + "/camera-intrinsics.txt"

    image = cv2.imread(image_file)
    depth = o3d.io.read_image(depth_file)
    pose = np.loadtxt(pose_file, dtype = np.float32)
    intrinsics = np.loadtxt(intrinsics_file, dtype = np.float32)

    return image, depth, pose, intrinsics

def load_ScanNet(path):
    num = os.path.splitext(os.path.basename(path))[0]
    image_file = path
    depth_file = os.path.dirname(os.path.dirname(path)) + '/depth/' + num + ".png"
    pose_file =  os.path.dirname(os.path.dirname(path)) + '/pose/' + num + ".txt"
    intrinsics_file = os.path.dirname(os.path.dirname(path)) + "/intrinsic/intrinsic_depth.txt"

    image = cv2.imread(image_file)
    depth = o3d.io.read_image(depth_file)
    pose = np.loadtxt(pose_file, dtype = np.float32)
    intrinsics = np.loadtxt(intrinsics_file, dtype = np.float32)

    crop_offset = 3  # 1296 -> 1290
    image = image[:, crop_offset:(image.shape[1] - crop_offset), :]
    image = cv2.resize(image, (640, 480))

    return image, depth, pose, intrinsics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type = str, default = "test")
    parser.add_argument("--progress_bar", default = True, action = "store_true")
    args = parser.parse_args()

    # Dataset configs to be decided
    default_cfg = get_cfg_defaults()
    default_cfg.defrost()
    align_cfg = default_cfg.ALIGN

    # Dataset Parameters
    dataset_cfg = default_cfg.DATASET
    dataset_cfg.num_views = 2
    data_loader = build_loader(dataset_cfg, split = args.split)

    # Registration
    register = ViGG(align_cfg)
    err_R, err_t, inlier_num, vis_recall, eps = [], [], [], [], []
    for batch in tqdm(data_loader, disable = not args.progress_bar, dynamic_ncols = True):
        path = [batch[f"path_{i}"] for i in range(dataset_cfg.num_views)]

        pose_pr, pose_gt = [], []
        for i in range(len(path[0])):
            # Read file with opencv and open3d
            if dataset_cfg.name == "ScanNet":
                img0, dep0, pose0, intrinsics = load_ScanNet(dataset_cfg.scannet_root + '/' + path[0][i])
                img1, dep1, pose1, _ = load_ScanNet(dataset_cfg.scannet_root + '/'  + path[1][i])
            elif dataset_cfg.name == "3DMatch":
                img0, dep0, pose0, intrinsics = load_3DMatch(dataset_cfg.rgbd_3d_root + '/' + path[0][i])
                img1, dep1, pose1, _ = load_3DMatch(dataset_cfg.rgbd_3d_root + '/'  + path[1][i])
            else:
                raise ValueError("Unknown Dataset '" + dataset_cfg.name + "'.")
            print(path[0][i])
            print(path[1][i])

            # Align
            result = register.align(img0, img1, dep0, dep1, intrinsics)
            res = result["trans"]
            pc0_corr = result["pc0_corr"]
            pc1_corr = result["pc1_corr"]
            visual_recall = result["visual_recall"]
            epsilon = result["epsilon"]

            # Check the inlier
            matches = []
            for i in range(pc0_corr.shape[0]):
                matches.append([i, i])
            inlier_num_filter = [InliersCheck(pc0_corr, pc1_corr, matches, pose0, pose1, thre_dis) for thre_dis in [0.1, 0.05, 0.025]]  # 10cm/5cm/2.5cm
            print(f"Inlier:     matches num:{len(matches)}  10cm inliers:{inlier_num_filter[0]}  5cm inliers:{inlier_num_filter[1]}  2.5cm inliers:{inlier_num_filter[2]}")
            inlier_num.append([len(matches), inlier_num_filter[0], inlier_num_filter[1], inlier_num_filter[2]])  # [match_num, 10cm_inlier_num, 5cm_inlier_num, 2.5cm_inlier_num]
            
            # Transformation of pc0 to pc1
            pose_pr.append(res)
            pose_gt.append(np.linalg.inv(pose1).dot(pose0))

            vis_recall.append(visual_recall)
            eps.append(epsilon)

        # Coumpute error
        print("---------------Registration Error---------------")
        for i in range(len(path[0])):
            RRE = compute_relative_rotation_error(pose_gt[i][0:3, 0:3], pose_pr[i][0:3, 0:3])
            RTE = compute_relative_translation_error(pose_gt[i][0:3, 3], pose_pr[i][0:3, 3])
            print(f"RRE: {RRE}, RTE: {RTE}")
            err_R.append(RRE)
            err_t.append(RTE)

    # Evaluate
    racc = np.array([len([x for x in err_R if x <= 2]), len([x for x in err_R if x <= 5]), len([x for x in err_R if x <= 10])], np.float64) / len(err_R) * 100
    tacc = np.array([len([x for x in err_t if x <= 0.05]), len([x for x in err_t if x <= 0.1]), len([x for x in err_t if x <= 0.25])], np.float64) / len(err_t) * 100
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
        if err_R[i] < 15 and err_t[i] < 0.3:
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
    os.mkdir(f"result/{timestamp}")
    np.save(f"result/{timestamp}/err_R.npy", np.array(err_R))
    np.save(f"result/{timestamp}/err_t.npy", np.array(err_t))
    np.save(f"result/{timestamp}/inlier_num.npy", np.array(inlier_num))
    np.save(f"result/{timestamp}/visual_recall.npy", np.array(vis_recall))
    np.save(f"result/{timestamp}/epsilon.npy", np.array(eps))
    with open(f"result/{timestamp}/cfg.pkl", "wb") as file:
        pickle.dump(align_cfg, file)