import torch
import torch.nn as nn
import config


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lamda_class = 1
        self.lamda_noobj = 10
        self.lamda_obj = 1
        self.lamda_box = 10

    def forward(self, prediction, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.bce(prediction[..., 0:1][noobj], target[..., 0:1][noobj])

        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(prediction[..., 1:3]), torch.exp(prediction[..., 3:5]) * anchors], dim=-1)
        ious = config.utils.intersection_over_unioin(target[..., 1:5][obj], box_preds[obj]).detach()
        object_loss = self.bce(prediction[..., 0:1][obj], ious * target[..., 0:1][obj])

        # Box loss
        prediction[..., 1:3] = self.sigmoid(prediction[..., 1:3])
        target[..., 3:5] = torch.log(
            target[..., 3:5] / anchors
        )
        box_loss = self.bce(prediction[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.entropy(prediction[..., 5:][obj], target[..., 5][obj].long())

        return (
                no_object_loss * self.lamda_noobj
                + object_loss * self.lamda_obj
                + box_loss * self.lamda_box
                + class_loss * self.lamda_class
        )
