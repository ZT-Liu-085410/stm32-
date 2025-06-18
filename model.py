import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class LivenessModel(nn.Module):
    def __init__(self, num_classes=2):
        super(LivenessModel, self).__init__()
        self.base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.base.classifier[1] = nn.Linear(self.base.last_channel, num_classes)

    def forward(self, x):
        return self.base(x)
