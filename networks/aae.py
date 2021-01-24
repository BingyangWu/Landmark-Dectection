import cv2
import os
import numpy as np

import torch
import torch.nn as nn

import config as cfg
import networks.invresnet
from networks.archs import D_net_gauss, Discriminator
from networks import resnet_ae, archs

from utils.nn import to_numpy, count_parameters, read_model, read_meta
from utils import vis
from landmarks import lmconfig as lmcfg
from skimage.metrics import structural_similarity as compare_ssim

cuda = torch.cuda.is_available()


def calc_acc(outputs, labels):
    assert outputs.shape[1] == 8
    assert len(outputs) == len(labels)
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels)
    acc = corrects.double() / float(outputs.size(0))
    return acc.item()


def pearson_dist(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    r = torch.sum(vx * vy, dim=0) / (
        torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0))
    )
    return 1 - r.abs().mean()


def resize_image_batch(X, target_size):
    resize = lambda im: cv2.resize(
        im, dsize=tuple(target_size), interpolation=cv2.INTER_CUBIC
    )
    X = X.cpu()
    imgs = [i.permute(1, 2, 0).numpy() for i in X]
    imgs = [resize(i) for i in imgs]
    tensors = [torch.from_numpy(i).permute(2, 0, 1) for i in imgs]
    return torch.stack(tensors)


def load_net(modelname):
    modelfile = os.path.join(cfg.SNAPSHOT_DIR, modelname)
    meta = read_meta(modelfile)
    input_size = meta.get("input_size", 256)
    output_size = meta.get("output_size", input_size)
    z_dim = meta.get("z_dim", 99)

    net = AAE(input_size=input_size, output_size=output_size, z_dim=z_dim)
    print("Loading model {}...".format(modelfile))
    read_model(modelfile, "saae", net)
    print("Model trained for {} iterations.".format(meta["total_iter"]))
    return net


class AAE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size=None,
        pretrained_encoder=False,
        input_channels=1,
        z_dim=99,
    ):

        super(AAE, self).__init__()

        assert input_size in [128, 256, 512, 1024]

        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.z_dim = z_dim

        self.Q = resnet_ae.resnet18(
            pretrained=pretrained_encoder,
            num_classes=self.z_dim,
            input_size=input_size,
            input_channels=input_channels,
            layer_normalization=cfg.ENCODER_LAYER_NORMALIZATION,
        )
        if cuda:
            self.Q = self.Q.cuda()

        decoder_class = networks.invresnet.InvResNet
        num_blocks = [cfg.DECODER_PLANES_PER_BLOCK] * 4
        self.P = decoder_class(
            networks.invresnet.InvBasicBlock,
            num_blocks,
            input_dims=self.z_dim,
            output_size=output_size,
            output_channels=input_channels,
            layer_normalization=cfg.DECODER_LAYER_NORMALIZATION,
            spectral_norm=cfg.DECODER_SPECTRAL_NORMALIZATION,
        )
        if cuda:
            self.P = self.P.cuda()

        self.D_z = D_net_gauss(self.z_dim)
        if cuda:
            self.D_z = self.D_z.cuda()
        self.D = Discriminator()
        if cuda:
            self.D = self.D.cuda()

        available_gpus = [0, 1, 2, 3]
        self.Q = torch.nn.DataParallel(self.Q, device_ids=available_gpus)
        # self.P = networks.invresnet.InvResNetParallel(self.P, device_ids=available_gpus)
        self.D_z = torch.nn.DataParallel(self.D_z, device_ids=available_gpus)
        self.D = torch.nn.DataParallel(self.D, device_ids=available_gpus)

        print("Trainable params Q: {:,}".format(count_parameters(self.Q)))
        print("Trainable params P: {:,}".format(count_parameters(self.P)))
        print("Trainable params D_z: {:,}".format(count_parameters(self.D_z)))
        print("Trainable params D: {:,}".format(count_parameters(self.D)))

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None

    def z_vecs(self):
        return [to_numpy(self.z)]

    def forward(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        self.landmark_heatmaps = None
        if outputs.shape[1] > 3:
            self.landmark_heatmaps = outputs[:, 3:]
        return outputs[:, :3]



def draw_results(X_resized, X_recon, levels_z=None, landmarks=None, landmarks_pred=None,
                 cs_errs=None, ncols=15, fx=0.5, fy=0.5, additional_status_text=''):

    clean_images = True
    if clean_images:
        landmarks=None

    nimgs = len(X_resized)
    ncols = min(ncols, nimgs)
    img_size = X_recon.shape[-1]

    disp_X = vis.to_disp_images(X_resized, denorm=True)
    disp_X_recon = vis.to_disp_images(X_recon, denorm=True)

    # reconstruction error in pixels
    l1_dists = 255.0 * to_numpy((X_resized - X_recon).abs().reshape(len(disp_X), -1).mean(dim=1))

    # SSIM errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        ssim[i] = compare_ssim(disp_X[i], disp_X_recon[i], data_range=1.0, multichannel=True)

    landmarks = to_numpy(landmarks)
    cs_errs = to_numpy(cs_errs)

    text_size = img_size/256
    text_thickness = 2

    #
    # Visualise resized input images and reconstructed images
    #
    if landmarks is not None:
        disp_X = vis.add_landmarks_to_images(disp_X, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)

    if landmarks_pred is not None:
        disp_X = vis.add_landmarks_to_images(disp_X, landmarks_pred, color=(1, 0, 0))
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, landmarks_pred, color=(1, 0, 0))

    if not clean_images:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, l1_dists, format_string='{:.1f}',
                                           size=text_size, thickness=text_thickness)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2, size=text_size, thickness=text_thickness)
        if cs_errs is not None:
            disp_X_recon = vis.add_error_to_images(disp_X_recon, cs_errs, loc='bl-2', format_string='{:>4.2f}',
                                                   vmax=0.0, vmin=0.4, size=text_size, thickness=text_thickness)

    # landmark errors
    lm_errs = np.zeros(1)
    if landmarks is not None:
        try:
            from landmarks import lmutils
            lm_errs = lmutils.calc_landmark_nme_per_img(landmarks, landmarks_pred)
            disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs, loc='br', format_string='{:>5.2f}', vmax=15,
                                                   size=img_size/256, thickness=2)
        except:
            pass

    img_input = vis.make_grid(disp_X, nCols=ncols, normalize=False)
    img_recon = vis.make_grid(disp_X_recon, nCols=ncols, normalize=False)
    img_input = cv2.resize(img_input, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    img_recon = cv2.resize(img_recon, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    img_stack = [img_input, img_recon]

    #
    # Visualise hidden layers
    #
    VIS_HIDDEN = False
    if VIS_HIDDEN:
        img_z = vis.draw_z_vecs(levels_z, size=(img_size, 30), ncols=ncols)
        img_z = cv2.resize(img_z, dsize=(img_input.shape[1], img_z.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_stack.append(img_z)

    cs_errs_mean = np.mean(cs_errs) if cs_errs is not None else np.nan
    status_bar_text = ("l1 recon err: {:.2f}px  "
                       "ssim: {:.3f}({:.3f})  "
                       "lms err: {:2} {}").format(
        l1_dists.mean(),
        cs_errs_mean,
        1 - ssim.mean(),
        lm_errs.mean(),
        additional_status_text
    )

    img_status_bar = vis.draw_status_bar(status_bar_text,
                                         status_bar_width=img_input.shape[1],
                                         status_bar_height=30,
                                         dtype=img_input.dtype)
    img_stack.append(img_status_bar)

    return np.vstack(img_stack)

def vis_reconstruction(
    net,
    inputs,
    landmarks=None,
    landmarks_pred=None,
    pytorch_ssim=None,
    fx=0.5,
    fy=0.5,
    ncols=10,
):
    net.eval()
    cs_errs = None
    with torch.no_grad():
        X_recon = net(inputs)

        if pytorch_ssim is not None:
            cs_errs = np.zeros(len(inputs))
            for i in range(len(cs_errs)):
                cs_errs[i] = (
                    1
                    - pytorch_ssim(
                        inputs[i].unsqueeze(0), X_recon[i].unsqueeze(0)
                    ).item()
                )

    inputs_resized = inputs
    landmarks_resized = landmarks
    if landmarks is not None:
        landmarks_resized = landmarks.cpu().numpy().copy()
        landmarks_resized[..., 0] *= inputs_resized.shape[3] / inputs.shape[3]
        landmarks_resized[..., 1] *= inputs_resized.shape[2] / inputs.shape[2]

    return draw_results(
        inputs_resized,
        X_recon,
        net.z_vecs(),
        landmarks=landmarks_resized,
        landmarks_pred=landmarks_pred,
        cs_errs=cs_errs,
        fx=fx,
        fy=fy,
        ncols=ncols,
    )
