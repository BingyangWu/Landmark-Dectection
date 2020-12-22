import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from utils.utils import norm2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(BasicBlock, self).__init__()
        self.layer_norm = layer_normalization
        self.downsample = downsample
        self.stride = stride

        self.net = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(layer_normalization)(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(layer_normalization)(planes)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.net(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, layer_normalization='batch'):
        super(Bottleneck, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            norm2d(layer_normalization)(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(layer_normalization)(planes),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            norm2d(layer_normalization)(planes * 4)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.net(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, input_size=256, input_channels=3, layer_normalization='batch'):
        super(ResNet, self).__init__()
        self.input_size = input_size
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer_norm = layer_normalization
        self.bn1 = norm2d(layer_normalization)(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.x2 = None

        self.with_additional_layers = True
        if input_size == 256 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        elif input_size == 512 and self.with_additional_layers:
            self.layer0 = self._make_layer(block, 64, layers[0])
            self.layer01 = self._make_layer(block, 64, layers[0], stride=2)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(self.layer_norm)(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, layer_normalization=self.layer_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_normalization=self.layer_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.input_size > 128 and self.with_additional_layers:
            x = self.layer0(x)
            if self.input_size > 256:
                x = self.layer01(x)

        self.x1 = self.layer1(x)
        self.x2 = self.layer2(self.x1)
        x = self.layer3(self.x2)
        x = self.layer4(x)
        self.ft = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'), strict=False)
        except RuntimeError or KeyError as e:
            print(e)
    return model