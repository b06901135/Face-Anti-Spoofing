import torch
import torch.nn as nn
import torchvision.models as models


# video models
class ResNet3D(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.video.r3d_18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


class ResNetMC3(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.video.mc3_18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


class ResNetR21D(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.video.r2plus1d_18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


# image models
class ResNet18(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.resnet18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


class ResNet50(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.resnet50(pretrained=pretrained)
        self.net.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        return self.net(x)


class Vgg11(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.vgg11_bn(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(4096, out_dim)

    def forward(self, x):
        return self.net(x)


class Vgg16(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.vgg16_bn(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(4096, out_dim)

    def forward(self, x):
        return self.net(x)


class Vgg19(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.vgg19_bn(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(4096, out_dim)

    def forward(self, x):
        return self.net(x)
