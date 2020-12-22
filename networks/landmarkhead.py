from torch import nn

class LandmarkHead(nn.Module):
    def __init__(self, block, layers, output_size=128, output_channels=68, layer_normalization='none', start_layer=2):
        super(LandmarkHead, self).__init__()
        self.layer_normalization = layer_normalization
        self.fc_landmarks = None
        self.output_size = output_size
        self.output_channels = output_channels
        self.start_layer = start_layer
        self.fc = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1, bias=False)


class LandmarkHeadV2(LandmarkHead):
    def __init__(self,block, layers, **params):
        super(LandmarkHeadV2, self).__init__(block, layers, **params)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, P):
        x = P.x1
        x = self.conv1(x)

        x = P.layer2(x)
        x = self.conv2(x)

        x = P.layer3(x)
        x = self.conv3(x)

        x = P.layer4(x)
        x = self.conv4(x)
        return self.fc(x)