import torch
import torch.nn as nn
import torchvision.models as models


class ResNet3D(nn.Module):
    def __init__(self, out_dim=5, pretrained=True):
        super().__init__()
        self.net = models.video.r3d_18(pretrained=pretrained)
        self.net.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = ResNet3D()
    print(model)

    fake_x = torch.ones((8, 3, 10, 112, 112))
    y = model(fake_x)
    print(y.size())
