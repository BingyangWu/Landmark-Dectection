import os
import cv2
import numpy as np

import torch

from utils import nn, cropping
from utils.vis import to_disp_image, add_landmarks_to_images

from landmarks import fabrec
from torchvision import transforms as tf
from landmarks import lmvis
from ml_utilities.transform import Nothing, ToTensor, CenterCrop
import config as cfg
from datasets.handdataset import HandDataset


def load_image(im_dir, fname, channels, size):
    from skimage import io

    img_path = os.path.join(im_dir, fname)
    img = io.imread(img_path)
    if img is None:
        raise IOError("\tError: Could not load image {}!".format(img_path))
    if len(img.shape) >= 3 and img.shape[2] == 4:
        print(fname, "converting RGBA to RGB...")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if channels == 3 and (len(img.shape) == 2 or img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if channels == 1 and len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    assert len(img.shape)==3, "invalid syntax: {}".format(fname)
    img = img.transpose(2,0,1)
    img = img.astype(np.float32)
    assert img.shape[0] in (1, 3)
    return img



if __name__ == "__main__":

    model = "./demo"
    net = fabrec.load_net(model, num_landmarks=17)
    net.eval()

    im_dir = "./images"
    img0 = "3.png"

    with torch.no_grad():

        img = load_image(im_dir, img0, channels=1, size=(256,256))
        img /= 256
        img = ToTensor()(img)
        img = cfg.RHPE_NORMALIZER(img)
        img = torch.unsqueeze(img, 0)
        print(img, img.shape)

        X_recon, lms_in_crop, X_lm_hm = net.detect_landmarks(img)
        outputs = add_landmarks_to_images(img, lms_in_crop, skeleton=HandDataset.SKELETON, denorm=True, draw_wireframe=True, color=(0,255,255))
        X_recon = X_recon[0, :, :, :]
        X_recon = to_disp_image(X_recon, True)
        cv2.imwrite("images/outputs.jpg", outputs[0])
        cv2.imwrite("images/reconstruct.jpg", X_recon)
