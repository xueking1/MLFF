# Muti-scale Lidar-Camera Feature Fusion for 3D Object Detection

MLFF is a two-stage 3D object detection method.Meanwhile.It is efficiency and accuracy stereo 3D object detection method for autonomous driving.

You can download the MLFF program files from the following website. [xueking1/MLFF · GitHub](https://github.com/xueking1/MLFF)

## Introduction

Muti-scale Lidar-Camera Feature Fusion method based on muti-scale and muti-source fusion strategy is designed in this paper, which is called MLFF(Muti-scale Lidar-Camera Feature Fusion for 3D Object Detecion). Multi-source features refer to image semantic features, voxel features and point features, and try to exploit the advantages of three data formats in one fusion algorithm at the same time.The whole network consists of three main modules: (1)Image stream ,(2) LiDAR stream , and (3) Voxel stream. Given an original feature map, the object point probability estimation and multi-scale feature map are obtained by image feature encoder. Then, certain point clouds are sampled to get the set of key points, and voxel features and 3D BEV feature map are obtained by voxel stream. Finally, the multi-scale feature map and voxel features are output to the LiDAR stream, which is fused with the key point set to obtain the multi-feature fused key points. The global features are obtained by combining the image, LiDAR and voxel branch to predict the detection results.

## Requirements

- Linux (tested on Ubuntu 18.04)
- Python 3.7+
- PyTorch 1.6
- pcdet0.5.2

## Installation

Refer to the following four sections to install MLFF.

### a.OpenPCDdet

Refer to  [GitHub - open-mmlab/OpenPCDet at v0.5.2](https://github.com/open-mmlab/OpenPCDet/tree/v0.5.2) and README-OpenPCDet.md to install OpenPCDet. 

## File Structure

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
MFAF
├─configs
│  ├─MFAF    <-- MV3D net related source code 
│  │      mfaf_kitti-3d-3class.py
├─data     <-- all data is stored here.
│  ├─kitti
│  │  │  kitti_dbinfos_train.pkl
│  │  │  kitti_infos_test.pkl
│  │  │  kitti_infos_train.pkl
│  │  │  kitti_infos_trainval.pkl
│  │  │  kitti_infos_val.pkl
│  │  │  Readme.txt
│  │  ├─.ipynb_checkpoints
│  │  ├─gt_database
│  │  ├─ImageSets
│  │  ├─testing
│  │  │  ├─calib & velodyne & image_2
│  │  ├──training
│  │  │  ├──calib & velodyne & label_2 & image_2 & (optional: planes)
├─demo
├─docker
├─docs
├─mmdet3d
├─requirements
├─resources
├─tests
├─tools
    │  create_data.py
    │  create_data.sh
    │  dist_test.sh
    │  dist_train.sh
    │  slurm_test.sh
    │  slurm_train.sh
    │  test.py
    │  train.py    <--- training the whole network. 
    │  update_data_coords.py
    │  update_data_coords.sh
    ├─analysis_tools
    ├─data_converter
    ├─deployment
    ├─misc
    └─model_converters
│  README.md
│  requirements.txt
│  setup.cfg
│  setup.py
```

## Modification needed to run

Follow Installation. After installing the environment, perform the following steps:

a.Training the KITTI dataset.

First you should set the data path in kitti_dataset.yaml

```python
DATA_PATH: ‘…/data/kitti’
```

You can train by running the following code

```python
python train.py --cfg_file cfgs/kitti_models/mfaf.yaml
```

b.Testing

```
python test.py --cfg_file cfgs/kitti_models/mfaf.yaml --batch_size 1 --ckpt /root/PointCloudDet3D/output/kitti_models/mfaf/default/ckpt/checkpoint_epoch_1.pth --save_to_file
```

## Some other readme.md files inside this repo

- README-OpenPCDet.md:How to install OpenPCDet.
