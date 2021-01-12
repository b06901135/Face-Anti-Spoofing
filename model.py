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


class AlexNet(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.alexnet(pretrained=pretrained)
        self.net.classifier[6] = nn.Linear(4096, out_dim)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = models.alexnet()
    print(model)

    # fake_x = torch.ones((8, 3, 10, 112, 112))
    # y = model(fake_x)
    # print(y.size())
