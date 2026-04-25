import torch
import torch.nn as nn
import timm


class DetectorModel(nn.Module):
    def __init__(self, model_name: str = "tf_efficientnetv2_s", dropout_rate: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
