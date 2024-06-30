import utils
import json
import os
import math
import sys

import argparse
from typing import Any, Optional

from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2.functional import get_size
from torchvision.transforms import v2 as T
from torchvision.models._utils import _ovewrite_value_param
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.boxes import masks_to_boxes

sys.path.append(os.path.abspath('../eval'))
from model import *

def maskrcnn_resnet50_fpn(
    *,
    weights = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> TwoHeadsMaskRCNN:
    weights = MaskRCNN_ResNet50_FPN_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = TwoHeadsMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if weights == MaskRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
    

def get_model_instance_segmentation(num_classes, num_second_classes):
    # load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(weights="DEFAULT", max_size=256, min_size=128, box_detections_per_img=15)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = TwoHeadsFastRCNNPredictor(in_features, num_classes, num_second_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


class VIMADetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples, label, transforms):
        self.root = root # image root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.samples = json.load(open(samples))
        self.label = label

    def __getitem__(self, idx):
        # load images and masks
        sample_info = self.samples[idx]
        img_path = os.path.join(self.root, sample_info['image_path'])
        mask_path = os.path.join(self.root, sample_info['mask_path'])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        # obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = torch.LongTensor([i['id'] for i in sample_info['object']])
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.LongTensor([self.label['color'].index(i['color']) * 100 + self.label['cls'].index(i['cls']) for i in sample_info['object']])
        # labels = torch.LongTensor([self.label['color'].index(i['color']) for i in sample_info['object']])
        
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)
    
    
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda')

    # our dataset has two classes only - background and person
    cls_data = json.load(open(args.classfile))
    num_classes = len(cls_data['color']) #  + 
    num_second_classes = len(cls_data['cls'])
    
    # use our dataset and defined transformations
    dataset = VIMADetectionDataset(args.image_folder, args.train, cls_data, get_transform(train=True))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=utils.collate_fn
    )
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, num_second_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = args.epoch

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        torch.save(model, os.path.join(args.output, f'{args.train.split("/")[-1][:-5]}-bs{args.batch_size}-ep{epoch + 1}.pth'))
        # update the learning rate
        lr_scheduler.step()
        
    print("That's it!")
    
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
