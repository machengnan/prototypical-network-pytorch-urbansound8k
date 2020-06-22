import torch.nn as nn
from torchvision import models
from torch.hub import  load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}


class mobilenet(models.mobilenet.MobileNetV2):
    def __init__(self, pretrained = True):

        super(mobilenet, self).__init__()
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=True)
            self.load_state_dict(state_dict)
            self.out_channels = 1600

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x.view(x.size(0), -1)

