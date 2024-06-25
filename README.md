<div align="center">   
  
# Towards Camera Open-set 3D Object Detection
</div>

> **Towards Camera Open-set 3D Object Detection**


# News
</br>


# Abstract
Recognizing the unknown objects from an image is a significant challenge for safely deploying 3D detectors in the real world. 
In previous works, camera 3D detectors are effective in identifying objects within certain categories.
However, due to the lack of training on the features of unknown objects, these detectors fail to generate detection responses when encountering them.
To mitigate the issue, we propose the LiDAR objectness-guided unknown discovery framework to generate pseudo-labels for unknown objects.
Our key idea is selecting potential unknown objects from the 3D region proposals. 
We make two improvements to our framework: First, a novel LiDAR Region Proposal Network (RPN) informed by geometric cues, designed to discover unknown objects more effectively. 
Second, a cross-modal pseudo-labels selecting module leverages both the objectness scores from the LiDAR RPN and the feature attention responses from the camera detector to more accurately identify suitable unknown objects from 3D region proposals.
Experiments on nuScenes and KITTI datasets demonstrate the effectiveness of our framework for camera 3D detectors to identify unknown objects even improving their detecting performance on known ones.


# Methods
![method](figs/arch.png "model arch")


# Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

# Model Zoo

<!-- | Backbone | Method | Lr Schd | NDS| mAP|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| R50 | BEVFormer-tiny_fp16 | 24ep | 35.9|25.7 | - |[config](projects/configs/bevformer_fp16/bevformer_tiny_fp16.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-small | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-base | 24ep | 51.7|41.6 |28500M |[config](projects/configs/bevformer/bevformer_base.py) | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.log) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t1-base | 24ep | 42.6 | 35.1 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t1-base | 48ep | 43.9 | 35.9 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-48ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t1 | 24ep | 45.3 | 38.1 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t1 | 48ep | 46.5 | 39.5 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-48ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t2 | 24ep | 51.8 | 42.0 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-24ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t2 | 48ep | 52.6 | 43.1 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) |
| [R50](https://pan.baidu.com/s/1Jh5Aq2YwcD6tdj7Sl5BB3g?pwd=5rij)  | BEVformerV2-t8 | 24ep | 55.3 | 46.0 | 40392M |[config](projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py) | [model/log](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv) | -->

# Catalog
- [next] 3D Detection checkpoints
- [next] 3D Detection code
- [todo] Initialization


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```

```

# Acknowledgement

Many thanks to these excellent open source projects:
- [bevformer](https://github.com/fundamentalvision/BEVFormer) 
- [detr3d](https://github.com/WangYueFt/detr3d) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

