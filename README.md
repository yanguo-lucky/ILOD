# 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection

This repo is the official implementation of our paper: <br>
**Improving Low-Light Object Detection via Domain Adaptation and Image Enhancement**<br>
the visual computer <br>

<p align="center">
<img src="ilod.jpg" width="95%">
</p>


# Installation

## Prerequisites

- Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- [Detectron2 == 0.6](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)


## Dataset download
1. Download the datasets (BDD100K / SHIFT)

2. Split BDD100K and SHIFT into day and night labels using dataset information. Convert BDD100K and SHIFT labels to coco format. Alternatively, you can download our split (https://www.dropbox.com/scl/fo/258uzp6i0dz17zsj234r6/h?dl=0&rlkey=kb6brfk1oqc1ddsa3ulz8v9ei).

3. Organize the dataset with the following format

```shell
2pcnet/
└── datasets/
    └── bdd100k/
        ├── train/ 
            ├── img00001.jpg
            ├──...
        ├── val/ 
            ├── img00003.jpg
            ├──...
        ├── train_day.json
        ├── train_night.json
        ├── val_night.json
    └── shift/
        ├── train/ 
            ├── folder1
            ├──...
        ├── val/ 
            ├── folder1
            ├──...
        ├── train_day.json
        ├── train_night.json
        ├── val_night.json

    
```

# Training

```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/faster_rcnn_R50_bdd100k.yaml\
      OUTPUT_DIR output/bdd100k
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 4 \
      --config configs/faster_rcnn_R50_bdd100k.yaml MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --config configs/faster_rcnn_R50_bdd100k.yaml \
      MODEL.WEIGHTS <your weight>.pth
```
# Acknowledgements
Code is adapted from [Detectron2](https://github.com/facebookresearch/detectron2) and [2pcnet](https://github.com/mecarill/2pcnet).
