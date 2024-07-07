# P2P 

The official implementation of the paper: 

**P2P: Part-to-Part Motion Cues Guide a Strong Tracking Framework for LiDAR Point Clouds**

>  [Jiahao Nie](https://scholar.google.com/citations?user=R0uN6pcAAAAJ&hl=zh-CN), [Fei Xie](https://scholar.google.com/citations?user=GbGmwiwAAAAJ&hl=zh-CN&oi=ao), [Sifan Zhou](https://scholar.google.com/citations?user=kSdqoi0AAAAJ&hl=zh-CN&oi=ao), [Xueyi Zhou](https://scholar.google.com/citations?hl=zh-CN&user=YflJMZcAAAAJ), [Dong-Kyu Chae](https://scholar.google.com/citations?hl=zh-CN&user=cUkDvwQAAAAJ), [Zhiwei He](https://scholar.google.com/citations?user=OZkiufUAAAAJ&hl=zh-CN&oi=ao).
>
> 📜 [[technical report](https://arxiv.org/abs/2407)], 🤗 [[model weights](https://drive.google.com/drive/folders/xx)]

<div id="top" align="center">
<p align="center">
<img src="figures/framework_comparison.png" width="1000px" >
</p>
</div>

## 🔥 Highlights

**P2P** is a strong tracking framework for 3D SOT on LiDAR point clouds that have:

- *Elegant tracking pipeline*.
- *SOTA performance on KITTI, NuScenes and Waymo*.
- *High efficiency*.

<div id="top" align="center">
<p align="center">
<img src="figures/kitti.jpg" width="1000px" >
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src="figures/nuscenes.jpg" width="1000px" >
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src="figures/waymo.jpg" width="1000px" >
</p>
</div>

## 📢 News

> [!IMPORTANT]
> If you have any question for our codes or model weights, please feel free to concat me at jhnie@hdu.edu.cn.

- **[2024/07/07]** We released the installation, training, and testing details.
- **[2024/07/07]** We released our [paper](https://arxiv.org/abs/2407) on arXiv.
- **[2024/07/06]** We released the implementation of our model.

## 📋 TODO List

- [ ] All caterogy model weights trained on KITTI, Nuscenes.

## 🕹️ Getting Started

- [Installation](https://github.com/OpenDriveLab/Vista/blob/main/docs/INSTALL.md)

- [Data Preparation](https://github.com/OpenDriveLab/Vista/blob/main/docs/DATA.md)

- [Training](https://github.com/OpenDriveLab/Vista/blob/main/docs/TRAINING.md)

- [Testing](https://github.com/OpenDriveLab/Vista/blob/main/docs/TESTING.md)


## ❤️ Acknowledgement

Our implementation is based on [Open3DSOT](https://github.com/Ghostish/Open3DSOT) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Thanks for their great open-source work!

## ⭐ Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{p2p,
 title={P2B: Part-to-Part Motion Cues Guide a Strong Tracking Framework for LiDAR Point Clouds},
 year={2024}
}


## Introduction

<p align="justify">3D single object tracking (SOT) methods based on appearance matching has long suffered from insufficient appearance information incurred by incomplete, textureless and semantically deficient LiDAR point clouds. While motion paradigm exploits motion cues instead of appearance matching for tracking, it incurs complex multi-stage processing and segmentation module. In this paper, we first provide in-depth explorations on motion paradigm, which proves that (i) it is feasible to directly infer target relative motion from point clouds across consecutive frames; (ii) fine-grained information comparison between consecutive point clouds facilitates target motion modeling. We thereby propose to perform part-to-part motion modeling for consecutive point clouds and introduce a novel tracking framework, termed P2P. The novel framework fuses each corresponding part information between consecutive point clouds, effectively exploring detailed information changes and thus modeling accurate target-related motion cues. Following this framework, we present P2P-point and P2P-voxel models, incorporating implicit and explicit part-to-part motion modeling by point- and voxel-based representation, respectively. Without bells and whistles, P2P-voxel sets a new state-of-the-art performance (~89%, 72% and 63% precision on KITTI, NuScenes and Waymo Open Dataset, respectively). Moreover, under the same point-based representation, P2P-point outperforms the previous motion tracker M2Track by 3.3% and 6.7% on the KITTI and NuScenes, while running at a considerably high speed of 107 Fps on a single RTX3090 GPU. 


## Setup
Here, we list the most important part of our dependencies

|Dependency|Version|
|---|---|
|python|3.9.0|
|pytorch|2.0.1|
|mmengine|0.7.4|
|mmcv|2.0.0|
|mmdet|3.0.0|
|mmdet3d|1.1.0|
|spconv|2.3.6|
|yapf|0.40.0|

## Dataset Preparation

### KITTI

+ Download the data for [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
+ Unzip the downloaded files.
+ Put the unzipped files under the same folder as following.
  ```
  [Parent Folder]
  --> [calib]
      --> {0000-0020}.txt
  --> [label_02]
      --> {0000-0020}.txt
  --> [velodyne]
      --> [0000-0020] folders with velodynes .bin files
  ```

### NuScenes

+ Download the dataset from the [download page](https://www.nuscenes.org/download)
+ Extract the downloaded files and make sure you have the following structure:
  ```
  [Parent Folder]
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	        -	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
  ```
>Note: We use the **train_track** split to train our model and test it with the **val** split. Both splits are officially provided by NuScenes. During testing, we ignore the sequences where there is no point in the first given bbox.


### Waymo Open Dataset

+ We follow the benchmark created by [LiDAR-SOT](https://github.com/TuSimple/LiDAR_SOT) based on the waymo open dataset. You can download and process the waymo dataset as guided by [LiDAR_SOT](https://github.com/TuSimple/LiDAR_SOT), and use our code to test model performance on this benchmark.
+ The following processing results are necessary
   ```
    [waymo_sot]
        [benchmark]
            [validation]
                [vehicle]
                    bench_list.json
                    easy.json
                    medium.json
                    hard.json
                [pedestrian]
                    bench_list.json
                    easy.json
                    medium.json
                    hard.json
        [pc]
            [raw_pc]
                Here are some segment.npz files containing raw point cloud data
        [gt_info]
            Here are some segment.npz files containing tracklet and bbox data 
    ```

## Quick Start

### Training

+ To train a model, you must specify the `.py` file. The `.py` file contains all the configurations of the dataset and the model. We provide `.py` files under the [configs](./configs) directory. 

>Note: Before running the code, you will need to edit the `.py` file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu training
    python train.py --config configs/voxel/kitti/car.py
    # multi-gpu training
    # you will need to edit the `train.py` file by setting the `config` argument
    ./dist_train.sh
    ```

### Testing

+ To test a trained model, specify the checkpoint location with `--resume_from` argument and set the `--phase` argument as `test`.

>Note: Before running the code, you will need to edit the `.py` file by setting the `path` argument as the correct root of the dataset.

    ```
    # single-gpu testing
    python test.py --config configs/voxel/kitti/car.py --load_from pretrained/voxel/kitti/car_73.64_85.68.pth
    # multi-gpu testing
    # you will need to edit the `test.py` file by setting the `config` and 'load_from' argument
    ./dist_test.sh
    ```

## Acknowledgement
This repo is built upon [Open3DSOT](https://github.com/Ghostish/Open3DSOT) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). We acknowledge these excellent implementations.
