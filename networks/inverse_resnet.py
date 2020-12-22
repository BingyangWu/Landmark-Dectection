import math
from torch import nn
from utils.utils import norm2d

class InvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, layer_normalization='batch',
                 with_spectral_norm=False):
        super(InvBasicBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=1, bias=False)
                if upsample is not None else 
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(layer_normalization)(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm2d(layer_normalization)(planes),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer_normalization = layer_normalization
        self.upsample = upsample
        self.stride = stride
        if with_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)

    def forward(self, x):
        residual = x

        out = self.net(x)

        if self.upsample is not None:
            residual = self.upsample(x)

        return self.relu(out + residual)

class InvResNet(nn.Module):

    def __init__(self, block, layers, output_size=256, output_channels=3, input_dims=99,
                 layer_normalization='none', spectral_norm=False):
        super(InvResNet, self).__init__()
        self.layer_normalization = layer_normalization
        self.with_spectral_norm = spectral_norm
        if self.with_spectral_norm:
            self.sn = nn.utils.spectral_norm
        else:
            self.sn = lambda x: x

        self.lin_landmarks = None
        self.inplanes = 512
        self.output_size = output_size
        self.output_channels = output_channels
        self.fc = nn.Linear(input_dims, 512)
        self.conv1 = self.sn(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0, bias=False))
        self.add_in_tensor = None

        self.norm = norm2d(layer_normalization)
        self.bn1 = self.norm(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        inplanes_after_layer1 = self.inplanes # self.inplanes gets changed in _make_layers
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        inplanes_after_layer2 = self.inplanes # self.inplanes gets changed in _make_layers
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.tanh = nn.Tanh()
        self.x2 = None
        if self.output_size == 256:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
        elif self.output_size == 512:
            self.layer5 = self._make_layer(block,  64, layers[3], stride=2)
            self.layer6 = self._make_layer(block,  64, layers[3], stride=2)

        self.lin = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_finetuning(self, batchsize):
        if self.add_in_tensor is None or self.add_in_tensor.shape[0] != batchsize:
            self._create_finetune_layers(batchsize)
        else:
            self._reset_finetune_layers()

    def _make_layer_down(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                self.sn(nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=4, stride=stride, padding=1, bias=False)),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample,
                            layer_normalization=self.layer_normalization,
                            with_spectral_norm=self.with_spectral_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.fc(x)
        x = x.view(x.size(0), -1, 1,1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.x0 = x

        x1 = self.layer1(x)
        self.x1 = x1
        self.x2 = self.layer2(x1)
        self.x3 = self.layer3(self.x2)
        self.x4 = self.layer4(self.x3)

        if self.output_size == 128:
            x = self.x4
        elif self.output_size == 256:
            x = self.layer5(self.x4)
            self.x5 = x
        elif self.output_size == 512:
            x = self.layer5(self.x4)
            x = self.layer6(x)

        x = self.lin(x)
        x = self.tanh(x)
        return x