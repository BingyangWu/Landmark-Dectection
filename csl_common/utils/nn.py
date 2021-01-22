import json
import os

import numpy as np
import torch

import config as cfg
cuda = torch.cuda.is_available()

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def to_numpy(ft):
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        return None


def to_image(m):
    img = to_numpy(m)
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    return img


def unsqueeze(x):
    if isinstance(x, np.ndarray):
        return x[np.newaxis, ...]
    else:
        return x.unsqueeze(dim=0)


def atleast4d(x):
    if len(x.shape) == 3:
        return unsqueeze(x)
    return x


def atleast3d(x):
    if len(x.shape) == 2:
        return unsqueeze(x)
    return x


class Batch:
    def __init__(self, data, n=None, gpu=True, eval=False):
        self.images = atleast4d(data["image"])

        self.eval = eval

        try:
            self.ids = data["id"]
            try:
                if self.ids.min() < 0 or self.ids.max() == 0:
                    self.ids = None
            except AttributeError:
                self.ids = np.array(self.ids)
        except KeyError:
            self.ids = None

        try:
            self.landmarks = atleast3d(data["landmarks"])
        except KeyError:
            self.landmarks = None

        try:
            self.filenames = data["filename"]
        except KeyError:
            self.filenames = None

        try:
            self.target_images = data["target_image"]
        except KeyError:
            self.target_images = None

        try:
            # self.face_weights = data['face_weights']
            self.lm_heatmaps = data["lm_heatmaps"]
            if len(self.lm_heatmaps.shape) == 3:
                self.lm_heatmaps = self.lm_heatmaps.unsqueeze(1)
        except KeyError:
            self.lm_heatmaps = None

        for each in data.keys():
            if each not in self.__dict__:
                self.__dict__[each] = data[each]

        if gpu:
            for k, v in self.__dict__.items():
                if v is not None:
                    try:
                        self.__dict__[k] = v.cuda()
                    except AttributeError:
                        pass

    def __len__(self):
        return len(self.images)


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def read_model(in_dir, model_name, model):
    filepath_mdl = os.path.join(in_dir, model_name + ".mdl")
    if not cuda:
        snapshot = torch.load(filepath_mdl, map_location=torch.device('cpu'))
    else:
        snapshot = torch.load(filepath_mdl)
    try:
        model.load_state_dict(snapshot["state_dict"], strict=False)
    except RuntimeError as e:
        print(e)


def read_meta(in_dir):
    with open(os.path.join(in_dir, "meta.json"), "r") as outfile:
        data = json.load(outfile)
    return data


def denormalize(img):
    return cfg.RHPE_NORMALIZER.denormalize(img)


def denormalized(tensor):
    # assert(len(tensor.shape[1] == 3)
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    return denormalize(t)
