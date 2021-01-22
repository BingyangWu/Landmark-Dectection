import numpy as np
from torchvision.transforms import functional as F
import numbers
import random
import cv2
import skimage.transform


class Transformer:
    def __init__(self):
        pass

    def transform(self, data):
        if isinstance(data, dict):
            data["image"] = self._transform(data["image"])
            return data
        else:
            return self._transform(data)

    def detransform(self, data):
        if isinstance(data, dict):
            data["image"] = self._detransform(data["image"])
            return data
        else:
            return self._detransform(data)

    def normalize(self, img):
        return self.transform(img)

    def denormalize(self, img):
        return self.detransform(img)

    def _transform(self, img):
        raise NotImplemented

    def __call__(self, img):
        return self.transform(img)

    def _detransform(self, img):
        raise NotImplemented

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__


class Transformers:
    def __init__(self):
        self.transformers = []

    def transform(self, img):
        for each in self.transformers:
            img = each(img)
        return img

    def __call__(self):
        return self.transform(img)

    def detransform(self, img):
        for each in self.transformers.reverse():
            img = each.detransform(img)
        return img


class Normalizer(Transformer):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _transform(self, img):
        img = (img - self.mean) / self.std
        return img

    def _detransform(self, img):
        img = img * self.std + self.mean
        return img

    def __repr__(self):
        return self.__class__.__name__ + "with mean: {}, std: {}".format(
            self.mean, self.std
        )


class Nothing(Transformer):
    def __init__(self):
        pass

    def _transform(self, img):
        return img

    def _detransform(self, img):
        return img


class CenterCrop(Transformer):
    """Like tf.CenterCrop, but works works on numpy arrays instead of PIL images."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __crop_image(self, img):
        t = int((img.shape[0] - self.size[0]) / 2)
        l = int((img.shape[1] - self.size[1]) / 2)
        b = t + self.size[0]
        r = l + self.size[1]
        return img[t:b, l:r]

    def _transform(self, sample):
        if isinstance(sample, dict):
            img, landmarks = sample["image"], sample["landmarks"]
            if landmarks is not None:
                landmarks[..., 0] -= int((img.shape[0] - self.size[0]) / 2)
                landmarks[..., 1] -= int((img.shape[1] - self.size[1]) / 2)
                landmarks[landmarks < 0] = 0
            sample.update({"image": self.__crop_image(img), "landmarks": landmarks})
            return sample
        else:
            return self.__crop_image(sample)

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class ToTensor(Transformer):
    """Convert ndarrays in sample to Tensors."""

    def _transform(self, sample):
        if isinstance(sample, dict):
            sample["image"] = self.__to_tensor(sample["image"])
            return sample
        else:
            return self.__to_tensor(sample)
            # return torch.from_numpy(sample)

    def __to_tensor(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        if image.shape[0] <= 3:
            image = image.transpose((1, 2, 0))
        return F.to_tensor(image)
