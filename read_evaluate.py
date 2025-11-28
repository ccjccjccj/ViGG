import numpy as np
import statistics
import pickle

timestamp = "2025-02-10_16-27-18"
visual_recall = np.load(f"result/{timestamp}/visual_recall.npy").tolist()
err_R = np.load(f"result/{timestamp}/err_R.npy").tolist()
err_t = np.load(f"result/{timestamp}/err_t.npy").tolist()
inlier_num = np.load(f"result/{timestamp}/inlier_num.npy").tolist()

with open(f"result/{timestamp}/cfg.pkl", "rb") as f:
    align_cfg = pickle.load(f)
print("================================")
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

print(f"        visual_recall: {sum(visual_recall) / len(visual_recall)}")
print(f" 10cm_inlier_num_Mean: {sum([inlier_num[i][1] / len(inlier_num) for i in range(len(inlier_num))])}")
print(f"  5cm_inlier_num_Mean: {sum([inlier_num[i][2] / len(inlier_num) for i in range(len(inlier_num))])}")
print(f"2.5cm_inlier_num_Mean: {sum([inlier_num[i][3] / len(inlier_num) for i in range(len(inlier_num))])}")

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

print("-------- For LATEX --------")
print(f"{racc[0]:.1f} / {racc[1]:.1f} / {racc[2]:.1f} & {statistics.median(err_R):.1f} & {tacc[0]:.1f} / {tacc[1]:.1f} / {tacc[2]:.1f} & {statistics.median(err_t) * 100:.1f} & {recall / len(err_R) * 100:.1f}")