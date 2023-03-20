import config
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader


class YoloDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            anchors,
            img_size=416,
            S=[13,26,52],
            C=20,
            transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transfrom = transform
        self.S = S
        self.C = C
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5
        

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
    
        img = np.array(Image.open(img_path).convert('RGB'))
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        
        if self.transfrom:
            augmentations = self.transfrom(image=img, bboxes=bboxes)
            bboxes = augmentations['bboxes']
            img = augmentations['image']

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in bboxes:
            iou_anchors = config.utils.iou_width_hight(torch.tensor(box[2:4]), self.anchors) 
            anchor_indicies = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]
            for anchor_idx in anchor_indicies:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
            
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return img, tuple(targets) 

if __name__ == '__main__':
    csv_path = '/home/maksm/Документы/datasets' + '/train.csv'
    data = YoloDataset(csv_path, '/home/maksm/Документы/datasets/images/', "/home/maksm/Документы/datasets/labels/", [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ])
    train = DataLoader(data)
    for x,y in train:
        print(y[0].shape)
        print(y[1].shape)
        print(y[2].shape)
    # data.__getitem__(1)
# torch.Size([3, 13, 13, 6])
# torch.Size([3, 26, 26, 6])
# torch.Size([3, 52, 52, 6])