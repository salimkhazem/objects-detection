import os
import numpy as np
import torch
from PIL import Image
import albumentations


class DetectDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        self.imgs = list(
            sorted(os.listdir(os.path.join(root, "Images_motor")))
        )  # Images_test
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks_motor"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images_motor", self.imgs[idx])
        mask_path = os.path.join(self.root, "Masks_motor", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        # print(boxes.shape, idx )
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # img = np.array(img)
        if self.transform is not None:
            img, target = self.transform(img, target)
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return img, target
        """if self.transform is not None:
            #img, target = self.transform(img, target)
            augmented = self.transform(image=img)
            img = augmented["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype = torch.float)
        #target = torch.tensor(target, dtype = torch.long)
        return img, target"""
