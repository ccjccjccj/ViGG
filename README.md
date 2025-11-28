# [WACV 2026] ViGG: Robust RGB-D Point Cloud Registration using Visual-Geometric Mutual Guidance

Code implementation of the paper:

ViGG: Robust RGB-D Point Cloud Registration using Visual-Geometric Mutual Guidance

## Requirements & Installation
- The code has been tested on following environment:  
Ubuntu 22.04  
CUDA 11.7  
python 3.9.18  

- To create a virtual environment and install the required dependences, please run:
```shell
conda create --name ViGG python=3.9.18
conda activate ViGG
pip install -r requirements.txt
```
- You need to install pytorch3d = 0.7.4 and MinkowskiEngine = 0.5.4 (for FCGF) separately, follow the official documents:  
"https://github.com/facebookresearch/pytorch3d"  
"https://github.com/NVIDIA/MinkowskiEngine"  

- The pretained model of LightGlue and FCGF will be downloaded automatically from following official urls when run the code:  
"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  
"https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"  
"https://node1.chrischoy.org/data/publications/fcgf/2019-08-19_06-17-41.pth"  
"https://node1.chrischoy.org/data/publications/fcgf/2019-07-31_19-37-00.pth"  

## Prepare 3DMatch/ScanNet Dataset
We evaluate in 3DMatch and ScanNet datasets, using the same dataset procedure as in previous RGB-D registration studies including PointMBF, UR&R and BYOC.

For the download and preparation procedure, please refer to [BYOC's instruction](https://github.com/mbanani/byoc/blob/main/docs/datasets.md).

After downloading the datasets, you need to update the paths in configs.

For our work, we only use test set for evaluation and no need with training, the pose matrices are only used for evaluation. 

## Test on 3DMatch/ScanNet Dataset
For test on 3DMatch/ScanNet dataset, please run:
```shell
python test.py
``` 
Results will be outputed to terminal when finished, and will also be saved in "result" folder with a timestamp for name.  

If you want to check the results again, you need to modify the timestamp in "read_evaluate.py", then run:
```shell
python read_evaluate.py
```

Test on 3DMatch/ScanNet dataset can be switched by modifying "_C.DATASET.name" in "config.py", and test of 20 frames apart can be made by changing "_C.DATASET.view_spacing".

## Prepare KITTI Dataset
KITTI odometry is an outdoor dataset, we use squences 8-10 for evaluation of our work. Please refer to [KITTI's official website](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), download "data_odometry_color" and "data_odometry_velodyne". Index, calib and ground truth are already included in "kitti_test.pkl". Then, you need to rearrange the folders like this:
```
<KITTI Root>
    |- 08
        |- color
            |- 000000.png
            |- 000001.png
            |- 000002.png
            ...
        |- velodyne
            |- 000000.bin
            |- 000001.bin
            |- 000002.bin
            ...
    |- 09
    |- 10
```
The files in "color" are the RGB images in "image_2" folder of "data_odometry_color". After that, you need to update the path in configs.

## Test on KITTI Dataset
For test on KITTI dataset, please run:
```shell
python test_KITTI.py
``` 
Results will be outputed to terminal when finished, and will also be saved in "result" folder with a timestamp for name.  

If you want to check the results again, you need to modify the timestamp in "read_evaluate_KITTI.py", then run:
```shell
python read_evaluate_KITTI.py
```

## Citation

```bibtex
@article{chen2025viggrobustrgbd,
      title={ViGG: Robust RGB-D Point Cloud Registration using Visual-Geometric Mutual Guidance}, 
      author={Congjia Chen and Shen Yan and Yufu Qu},
      journal={IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2026},
}
```