import torch.nn as nn
import torchvision.models as tv_models
from vit_pytorch.mobile_vit import MobileViT

from models.external.resnet import resnet18, resnet34
from models.external.vgg import vgg
from models.external.cnn import cnn

class ResNet(nn.Module):
    def __init__(self, model: str, num_classes: int):
        super().__init__()
        if model == 'ResNet18':
            self.model = resnet18(num_classes=num_classes)
        elif model == 'ResNet34':
            self.model = resnet34(num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class VGG(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = vgg(num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class MobileViT_IN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MobileViT(
            image_size = (256, 256),
            dims = [96, 120, 144],
            channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
            num_classes = 1000
            )
    def forward(self, x):
        return self.model(x)
    
class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = cnn(num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class IN_base_learner(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        if name == 'ResNet18_IN':
            self.model = tv_models.resnet18()
        elif name == 'ResNet34_IN':
            self.model = tv_models.resnet34()
        elif name == 'SqueezeNet_IN':
            self.model = tv_models.squeezenet1_1()
        elif name == 'DenseNet_IN':
            self.model = tv_models.densenet121()
        elif name == 'ShuffleNet_IN':
            self.model = tv_models.shufflenet_v2_x1_0()
        elif name == 'MobileNet_IN':
            self.model = tv_models.mobilenet_v3_small()
        elif name == 'MobileViT_IN':
            self.model = MobileViT_IN()
        else:
            raise ValueError(f'Unknown base learner: {name}')
    def forward(self, x):
        return self.model(x)
    