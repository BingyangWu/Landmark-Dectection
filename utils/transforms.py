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

class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample=False, fillcolor=0, keep_aspect=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.angle_range = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.angle_range = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale_range = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear_range = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear_range = shear
        else:
            self.shear_range = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.keep_aspect = keep_aspect

    # @staticmethod
    # def get_params(degrees, translate, scale_range, shears, img_size, keep_aspect):
    def get_params(self, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        if self.translate is not None:
            max_dx = self.translate[0] * img_size[0]
            max_dy = self.translate[1] * img_size[1]
            translations = (-np.round(random.uniform(-max_dx, max_dx)),
                            -np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if self.scale_range is not None:
            scale_x = random.uniform(self.scale_range[0], self.scale_range[1])
            if self.keep_aspect:
                scale_y = scale_x
            else:
                scale_y = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_x, scale_y = 1.0, 1.0

        if self.shear_range is not None:
            shear = random.uniform(self.shear_range[0], self.shear_range[1])
        else:
            shear = 0.0

        return angle, translations, (scale_x, scale_y), shear

    def _get_full_matrix(self, angle, translations, scales, shear, img_size):
        M = skimage.transform.AffineTransform(
            rotation=np.deg2rad(angle),
            translation=translations,
            shear=np.deg2rad(shear),
            scale=scales,
        )
        t = skimage.transform.AffineTransform(translation=-np.array(img_size[::-1])/2)
        return skimage.transform.AffineTransform(matrix=t._inv_matrix.dot(M.params.dot(t.params)))

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        else:
            img = sample

        angle, translations, scale, shear = self.get_params(img.shape[:2])
        M = self._get_full_matrix(angle, translations, scale, shear, img.shape[:2])
        img_new = transform_image(img, M)

        if isinstance(sample, dict):
            if landmarks is None:
                landmarks_new = None
            else:
                landmarks_new = transform_landmarks(landmarks, M).astype(np.float32)
            return {'image': img_new, 'landmarks': landmarks_new, 'pose': pose}
        else:
            return img_new

    def get_matrix(self, img_size):
        if isinstance(img_size, numbers.Number):
            self.degrees = (img_size, img_size)
        else:
            assert isinstance(img_size, (tuple, list)) and len(img_size) == 2, \
                "img_size should be a list or tuple and it must be of length 2."
        return self.get_params(img_size)

    def __repr__(self):
        s = f'{self.__class__.__name__}(degrees={self.angle_range}'
        if self.translate is not None:
            s += f', translate={self.translate}'
        if self.scale_range is not None:
            s += f', scale={self.scale_range}'
        if self.shear_range is not None:
            s += f', shear={self.shear_range}'
        if self.resample > 0:
            s += f', resample={self.resample}'
        if self.fillcolor != 0:
            s += f', fillcolor={self.fillcolor}'
        s += ')'
        return s

class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    lm_left_to_right_98 = {
        # outline
        0:32,
        1:31,
        2:30,
        3:29,
        4:28,
        5:27,
        6:26,
        7:25,
        8:24,

        9:23,
        10:22,
        11:21,
        12:20,
        13:19,
        14:18,
        15:17,
        16:16,

        #eyebrows
        33:46,
        34:45,
        35:44,
        36:43,
        37:42,
        38:50,
        39:49,
        40:48,
        41:47,

        #nose
        51:51,
        52:52,
        53:53,
        54:54,

        55:59,
        56:58,
        57:57,

        #eyes
        60:72,
        61:71,
        62:70,
        63:69,
        64:68,
        65:75,
        66:74,
        67:73,
        96:97,

        #mouth outer
        76:82,
        77:81,
        78:80,
        79:79,
        87:83,
        86:84,
        85:85,

        #mouth inner
        88:92,
        89:91,
        90:90,
        95:93,
        94:94,
    }

    lm_left_to_right_68 = {
        # outline
        0:16,
        1:15,
        2:14,
        3:13,
        4:12,
        5:11,
        6:10,
        7:9,
        8:8,

        #eyebrows
        17:26,
        18:25,
        19:24,
        20:23,
        21:22,

        #nose
        27:27,
        28:28,
        29:29,
        30:30,

        31:35,
        32:34,
        33:33,

        #eyes
        36:45,
        37:44,
        38:43,
        39:42,
        40:47,
        41:46,

        #mouth outer
        48:54,
        49:53,
        50:52,
        51:51,
        57:57,
        58:56,
        59:55,

        #mouth inner
        60:64,
        61:63,
        62:62,
        66:66,
        67:65,
    }

    # AFLW
    lm_left_to_right_21 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:16,
        13:15,
        14:14,
        17:19,
        18:18,
        20:20
    }

    # AFLW without ears
    lm_left_to_right_19 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:14,
        13:13,
        15:17,
        16:16,
        18:18
    }

    lm_left_to_right_5 = {
        0:1,
        2:2,
        3:4,
    }

    lm_left_to_right_38 = {
        # eye brows
        0: 5,
        1: 4,
        2: 3,

        # eyes
        12: 24,
        13: 23,
        14: 22,
        15: 21,
        16: 20,
        17: 27,
        18: 26,
        19: 25,

        # nose
        6: 6,
        7: 7,
        8: 8,
        9: 11,
        10: 10,

        # mouth
        28: 34,
        29: 33,
        30: 32,
        31: 31,
        36: 36,
        37: 37
    }

    # DeepFashion full body fashion landmarks
    lm_left_to_right_8 = {
        0:1,
        2:3,
        4:5,
        6:7,
    }

    def __init__(self, p=0.5):

        def build_landmark_flip_map(left_to_right):
            map = left_to_right
            right_to_left = {v:k for k,v in map.items()}
            map.update(right_to_left)
            return map

        self.p = p

        self.lm_flip_map_98 = build_landmark_flip_map(self.lm_left_to_right_98)
        self.lm_flip_map_68 = build_landmark_flip_map(self.lm_left_to_right_68)
        self.lm_flip_map_21 = build_landmark_flip_map(self.lm_left_to_right_21)
        self.lm_flip_map_19 = build_landmark_flip_map(self.lm_left_to_right_19)
        self.lm_flip_map_5 = build_landmark_flip_map(self.lm_left_to_right_5)
        self.lm_flip_map_8 = build_landmark_flip_map(self.lm_left_to_right_8)
        self.lm_flip_map_38 = build_landmark_flip_map(self.lm_left_to_right_38)


    def __call__(self, sample):
        if random.random() < self.p:
            if isinstance(sample, dict):
                img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
                # flip image
                flipped_img = np.fliplr(img).copy()
                # flip landmarks
                non_zeros = landmarks[:,0] > 0
                landmarks[non_zeros, 0] *= -1
                landmarks[non_zeros, 0] += img.shape[1]
                landmarks_new = landmarks.copy()
                if len(landmarks) == 21:
                    lm_flip_map = self.lm_flip_map_21
                elif len(landmarks) == 19:
                    lm_flip_map = self.lm_flip_map_19
                elif len(landmarks) == 68:
                    lm_flip_map = self.lm_flip_map_68
                elif len(landmarks) == 5:
                    lm_flip_map = self.lm_flip_map_5
                elif len(landmarks) == 98:
                    lm_flip_map = self.lm_flip_map_98
                elif len(landmarks) == 8:
                    lm_flip_map = self.lm_flip_map_8
                elif len(landmarks) == 38:
                    lm_flip_map = self.lm_flip_map_38
                else:
                    raise ValueError('Invalid landmark format.')
                for i in range(len(landmarks)):
                    landmarks_new[i] = landmarks[lm_flip_map[i]]
                # flip pose
                if pose is not None:
                    pose[1] *= -1
                return {'image': flipped_img, 'landmarks': landmarks_new, 'pose': pose}

            return np.fliplr(sample).copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
