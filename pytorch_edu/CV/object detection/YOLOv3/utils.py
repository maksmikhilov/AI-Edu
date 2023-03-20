import torch
import numpy as np
import os
import random
from dataset import YoloDataset
from torch.utils.data import DataLoader
import config


def intersection_over_unioin(boxes_label, boxes_pred, box_format='midpoint'):
    
    """
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    """
    

    if box_format == 'midpoint':
        box_pred_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        box_pred_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        box_pred_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
        box_pred_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2

        box_label_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3] / 2
        box_label_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4] / 2
        box_label_x2 = boxes_label[..., 0:1] + boxes_label[..., 2:3] / 2
        box_label_y2 = boxes_label[..., 1:2] + boxes_label[..., 3:4] / 2

    if box_format == 'corners':
        box_pred_x1 = boxes_pred[..., 0:1]
        box_pred_y1 = boxes_pred[..., 1:2]
        box_pred_x2 = boxes_pred[..., 2:3]
        box_pred_y2 = boxes_pred[..., 3:4]

        box_label_x1 = boxes_label[..., 0:1]
        box_label_y1 = boxes_label[..., 1:2]
        box_label_x2 = boxes_label[..., 2:3]
        box_label_y2 = boxes_label[..., 3:4]

    x1 = torch.max(box_label_x1, box_pred_x1)
    y1 = torch.max(box_label_y1, box_pred_y1)
    x2 = torch.max(box_label_x2, box_pred_x2)
    y2 = torch.max(box_label_y2, box_pred_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box_pred_area = abs((box_pred_x2 - box_pred_x1) * (box_pred_y2 - box_pred_y1))
    box_label_area = abs((box_label_x2 - box_label_x1) * (box_label_y2 - box_label_y1))

    return intersection / (box_pred_area + box_label_area - intersection + 1e-6)

def iou_width_hight(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes2[..., 0] + boxes1[..., 1] * boxes2[..., 1] - intersection
    return intersection / union


def non_max_supression(bboxes, iou_treshold, treshold, box_format='corners'):

    """
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    bboxes = [box for box in bboxes if box[1] > treshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop()

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_unioin(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            )
            < iou_treshold
        ]

        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(train_csv, test_csv, num_classes):

    IMAGE_SIZE = config.IMAGE_SIZE

    load_train = YoloDataset(
        csv_file=config.DATASET_DIR + train_csv,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        img_size=IMAGE_SIZE,
        S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        C=num_classes,
        transform=config.train_transforms
    )

    load_test = YoloDataset(
        csv_file=config.DATASET_DIR + test_csv,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        img_size=IMAGE_SIZE,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        C=num_classes,
        transform=config.test_transforms
    )

    train = DataLoader(
        load_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True, 
    )

    test = DataLoader(
        load_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
    )

    return train, test