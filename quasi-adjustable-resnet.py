import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_layers=4):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_layers, stride=1)
        self.layer2 = self._make_layer(128, num_layers, stride=2)
        self.layer3 = self._make_layer(256, num_layers, stride=2)
        self.layer4 = self._make_layer(512, num_layers, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, planes, num_layers, stride):
        layers = []
        layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        for i in range(num_layers-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x) + x
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        x = self.layer4(x) + x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
