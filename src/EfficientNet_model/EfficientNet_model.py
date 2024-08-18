import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class EfficientNet_Mod(nn.Module):
    def __init__(self):
        super(EfficientNet_Mod, self).__init__()
        self.model = efficientnet_b0(pretrained=False)

        self.model.features[0][0] = nn.Conv2d(in_channels=1,
                                              out_channels=32,
                                              kernel_size=(3,3),
                                              stride=(2,2),
                                              padding=(1,1),
                                              bias=False)
        def forward(self, x):
            return self.model(x)
