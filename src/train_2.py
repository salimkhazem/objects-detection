import os
from dataset import DetectDataset
from models import *
import utils
import torch
from tqdm import tqdm
from PIL import Image
from engine import train_one_epoch, evaluate

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T


def get_instance_segmentation_model(num_classes=2):  # (2)
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


dataset = DetectDataset("../input/", get_transform(train=True))
dataset_test = DetectDataset("../input/", get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:2500])  # :-50
dataset_test = torch.utils.data.Subset(dataset_test, indices[2500:])  # -50:

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=utils.collate_fn,
)


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# our dataset has two classes only - background and screw
num_classes = 2  # 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.001, momentum=0.9, weight_decay=0.0005  # 0.005
)  # 0.0005

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


num_epochs = 12

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate using scheduler
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# pick one image from the test set
for i in tqdm(range(5), position=0, leave=True):
    img, _ = dataset_test[i]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    s = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    v = Image.fromarray(prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy())
    s.save(f"../results/Valve_detection/{i}.png", format="png")
    # v.save(f"Mask_screw_test{i}.png", format="png")
