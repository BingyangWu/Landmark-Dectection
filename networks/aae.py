import torch.nn as nn

from networks.inverse_resnet import InvResNet, InvBasicBlock
from networks.discriminator import D_z, Discriminator
from networks.resnet import ResNet18

from utils.utils import to_numpy
import config
class AAE(nn.Module):
    def __init__(self, input_size, output_size=None, pretrained_encoder=False,
                 z_dim=99, cfg=config):

        super(AAE, self).__init__()

        assert input_size in [128, 256, 512, 1024]

        self.input_size = input_size
        if output_size is None:
            output_size = input_size
        self.z_dim = z_dim
        input_channels = 3 #?

        self.Q = ResNet18(pretrained=pretrained_encoder,
                          num_classes=self.z_dim,
                          input_size=input_size,
                          input_channels=input_channels,
                          layer_normalization=cfg.ENCODER_LAYER_NORMALIZATION
                        ).cuda()

        num_blocks = [cfg.DECODER_PLANES_PER_BLOCK] * 4
        self.P = InvResNet(InvBasicBlock,
                           num_blocks,
                           input_dims=self.z_dim,
                           output_size=output_size,
                           output_channels=input_channels,
                           layer_normalization=cfg.DECODER_LAYER_NORMALIZATION,
                           spectral_norm=cfg.DECODER_SPECTRAL_NORMALIZATION,
                        ).cuda()

        self.D_z = D_z(self.z_dim).cuda()
        self.D = Discriminator().cuda()

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None

    def z_vecs(self):
        return [to_numpy(self.z)]

    def forward(self, X):
        self.z = self.Q(X)
        out = self.P(self.z)
        self.heatmaps = out[:,3:] if out.shape[1] > 3 else None
        return out[:,:3]