# Train MaskRCNN on VIMA Dataset

This page will guide you to train your MaskRCNN object detector on VIMA datasets. If you haven't yet generate the dataset prefixed with `maskrcnn-train`, please refer to the last section of [convert_vima.ipynb](../datasets/convert_vima.ipynb).

## Quick Start

Usage: 
```
python3 train.py [-h] [--train TRAIN] [--output OUTPUT] [--classfile CLASSFILE] [--image_folder IMAGE_FOLDER] [--batch_size BATCH_SIZE] [--epoch EPOCH]
```
- `TRAIN`: Path to the training dataset (e.g. `../datasets/maskrcnn-train-8k-front.json`).
- `OUTPUT`: Directory where the trained models will be saved.
- `CLASSFILE`: File containing a list of classes in the dataset.
- `IMAGE_FOLDER`: Directory containing the dataset images.
- `BATCH_SIZE`: Training batch size (a default value of 16 requires 24GB of GPU RAM).
- `EPOCH`: Number of training epochs.

## Pretrained Models
| VIMA-0.8k | VIMA-8k | VIMA-80k |
|---|---|---|
| [Google Drive](https://drive.google.com/file/d/1y8u9Xs12gr_POFfseehgQ9PflL_BMbPy/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1Ej_KNGbRNa_9J8sHkUiHHnxdcKaMPnlh/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1dzZNYRQwNRCWb9ZQHW6_-RArPPVF4ojr/view?usp=drive_link) |
 

## Understanding the Model

The object detector here is a modified version of MaskRCNN, whose visual backbone is initialized from ResNet-50 trained on MS-COCO. 
This modified MaskRCNN is designed to produce two sets of labels: one for textures and another for shapes, optimizing it for specific traits in the VIMA dataset.

## Acknowledgements

This code has been adapted from the [Torch Vision tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). A big shout-out to the community for their invaluable contributions!
