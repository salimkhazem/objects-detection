import os 
from dataset_inference import DetectDataset
from models import *
import utils
import torch 
from tqdm import tqdm 
from PIL import Image, ImageDraw 
from engine import train_one_epoch, evaluate
import cv2 
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator 
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
import albumentations 
import time 

thresh = float(os.environ.get("THRESH"))


def get_instance_segmentation_model(num_classes=2): # 2
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
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
aug = albumentations.Compose(
            [
                #albumentations.Normalize(
                #    mean, std)
                albumentations.HorizontalFlip(p=0.2), 
                albumentations.VerticalFlip(p=0.2),
                albumentations.RandomCrop(255,255)
            ]
        )

dataset = DetectDataset('../input/', get_transform(train=True))
#dataset_test = DetectDataset('../input/',aug) #Using Albumentations (Must be trained)
dataset_test = DetectDataset('../input/',get_transform(train=False))
#test = DetectDataset("../input/")
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:900])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[1400:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=16, shuffle=False, num_workers=8,
    collate_fn=utils.collate_fn)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and screw
num_classes = 2 #2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
#model = model.load_state_dict(torch.load("../saved_models/best_model_occ.pth"))
#model.load_state_dict(torch.load("../saved_models/best_model_screw_rtx3090.pth")) # Saved model (Best model 28/06)
#model.load_state_dict(torch.load("../saved_models/best_model_screw_rtx3090.pth")) # Saved model  for screw detection (Best model Rtx 3090 28/06)
model.load_state_dict(torch.load("../saved_models/best_model_motor.pth"))
#model.load_state_dict(torch.load("../saved_models/best_model_aug.pth")) # Saved model (Best model 28/06)
model.eval()
# move model to the right device
model.to(device)




# pick one image from the test set
for i in tqdm(range(len(dataset_test)), position=0, leave=True): 
    img, _ = dataset_test[i]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    
    #print(len(prediction[0]['boxes']))
    s=Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    #cd v=Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    t=Image.fromarray(prediction[0]['boxes'].byte().cpu().numpy())
    #l=Image.fromarray(prediction[0]['labels'][0, 0].mul(255).byte().cpu().numpy())
    #m=Image.fromarray(prediction[0]['scores'][0, 0].mul(255).byte().cpu().numpy())
    img = ImageDraw.Draw(s)
    for j in range(len(prediction[0]['boxes'])) : 
        img.rectangle(prediction[0]['boxes'][j].cpu().numpy(),width=3 ,outline="red")
        #print("Image : ", i,prediction[0]['labels'][j], "Score : ", prediction[0]['scores'][j])
        #img.rectangle(prediction[0]['boxes'][1].cpu().numpy())
        


        if (prediction[0]['scores'][j]) > thresh:
            s.save(f"../results/Alpha_Motor_{i}.png", format="png")
        
    #v.save(f"Mask_test_{i}.png", format="png")
    #t.save(f"boxe_screw_test_{i}.png", format="png")
    #l.save(f"label_screw_test_{i}.png", format="png")
    #m.save(f"scores_screw_test_{i}.png", format="png")

