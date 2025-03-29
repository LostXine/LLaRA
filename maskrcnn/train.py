import os
import sys
import json
import math
import torch
import argparse
from typing import Any, Optional

from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import get_size
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.boxes import masks_to_boxes

import utils
sys.path.append(os.path.abspath('../eval'))
from model_maskrcnn import TwoHeadsMaskRCNN, TwoHeadsFastRCNNPredictor

def maskrcnn_resnet50_fpn(
    *,
    weights=None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> TwoHeadsMaskRCNN:
    """Create a custom Mask R-CNN model with a ResNet-50-FPN backbone."""
    weights = MaskRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91  # Default for COCO

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else torch.nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = TwoHeadsMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == MaskRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model


def get_transform(train: bool):
    """Return the image transformations to apply to dataset samples."""
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes: int, num_second_classes: int):
    """Get the instance segmentation model customized for dual classification heads."""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT", max_size=256, min_size=128, box_detections_per_img=15)

    # Replace classification and mask heads
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = TwoHeadsFastRCNNPredictor(in_features, num_classes, num_second_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


class VIMADetectionDataset(torch.utils.data.Dataset):
    """Custom Dataset for Instance Segmentation from VIMA annotated JSON."""

    def __init__(self, root, samples, label, transforms):
        self.root = root
        self.transforms = transforms
        self.samples = json.load(open(samples))
        self.label = label

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_path = os.path.join(self.root, sample_info['image_path'])
        mask_path = os.path.join(self.root, sample_info['mask_path'])

        img = read_image(img_path)
        mask = read_image(mask_path)

        obj_ids = torch.LongTensor([obj['id'] for obj in sample_info['object']])
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)

        labels = torch.LongTensor([
            self.label['color'].index(obj['color']) * 100 + self.label['cls'].index(obj['cls'])
            for obj in sample_info['object']
        ])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    """Train the model for one epoch."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if not math.isfinite(losses_reduced.item()):
            print(f"Loss is {losses_reduced.item()}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train(args):
    """Main training loop."""
    device = torch.device('cuda')

    cls_data = json.load(open(args.classfile))
    num_classes = len(cls_data['color'])
    num_second_classes = len(cls_data['cls'])

    dataset = VIMADetectionDataset(
        args.image_folder, args.train, cls_data, get_transform(train=True)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes, num_second_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epoch):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        save_path = os.path.join(
            args.output,
            f'{os.path.basename(args.train)[:-5]}-bs{args.batch_size}-ep{epoch + 1}.pth'
        )
        torch.save(model, save_path)
        lr_scheduler.step()

    print("Training complete!")


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../datasets/maskrcnn-train-8k-front.json')
    parser.add_argument('--output', type=str, default='../checkpoints')
    parser.add_argument('--classfile', type=str, default='../eval/classes.json')
    parser.add_argument('--image_folder', type=str, default='/mnt/dist/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=2)
    args = parser.parse_args()

    train(args)