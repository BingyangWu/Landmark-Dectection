import numpy as np
import cv2
from skimage import io

from datasets.imagedataset import ImageDataset
from landmarks import lmutils
from utils.vis import color_map, to_disp_image
from utils import transforms as csl_tf
from landmarks.lmvis import show_landmark_heatmaps
from landmarks import lmconfig as lmcfg
from torchvision import transforms as tf


class HandDataset(ImageDataset):
    NUM_LANDMARKS = 17
    ALL_LANDMARKS = range(NUM_LANDMARKS)
    SKELETON = (
        (1, 2),
        (2, 3),
        (3, 16),
        (4, 5),
        (5, 6),
        (6, 16),
        (7, 8),
        (8, 9),
        (9, 16),
        (10, 11),
        (11, 12),
        (12, 16),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
    )

    def __init__(
        self,
        return_landmark_heatmaps=False,
        landmark_sigma=9,
        align_face_orientation=False,
        image_size=256,
        **kwargs
    ):
        super().__init__(image_size=image_size, **kwargs)
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.landmark_sigma = landmark_sigma
        self.empty_landmarks = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.float32)
        if isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)
        print("successfully loaded {} images".format(len(self)))

    def _crop_landmarks(self, lms):
        return self.loader._cropper.apply_to_landmarks(lms)[0]

    def get_sample(self, filename, bb, landmarks):
        def transform(image, bb, image_size):
            return image

        try:
            image = cv2.imread(filename)
            if len(image.shape) == 3:
                image = image[:, :, 0]
            imgsize = image.shape
        except:
            raise IOError("\tError: Could not load image {}".format(filename))

        image = image[bb[1] : bb[3], bb[0] : bb[2]]
        try:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        except:
            raise IOError(
                "An error occurred when resizing image {} with size {}.".format(
                    filename, image.shape
                )
            )
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image /= 256

        sample = {"image": image, "landmarks": landmarks}

        sample = self.crop_to_tensor(sample)

        if self.return_landmark_heatmaps:
            heatmap_size = lmcfg.HEATMAP_SIZE
            _landmarks = []
            for each in landmarks:
                each[0] -= bb[0]
                each[1] -= bb[1]
                each[0] = int(each[0] * self.image_size[0] / (bb[2] - bb[0]))
                each[1] = int(each[1] * self.image_size[1] / (bb[3] - bb[1]))
                _landmarks.append(each)
            landmarks = np.array(_landmarks)
            lm_heatmaps = lmutils.create_landmark_heatmaps(
                landmarks,
                self.landmark_sigma,
                self.ALL_LANDMARKS,
                lmcfg.HEATMAP_SIZE,
                self.image_size[0],
            )
            sample.update({"lm_heatmaps": lm_heatmaps})

        sample.update({"filename": filename})

        return sample

    @property
    def demo(self):
        from random import randint

        l = len(self)
        ind = randint(0, l - 1)

        item = self[ind]
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                print(key, "np.ndarray with size", value.shape)
            else:
                print(key, value)
            try:
                print("\tshape:", value.shape)
            except:
                pass

        temp_folder = "./temp/"
        cv2.imwrite(
            temp_folder + "loaded_image.jpg", to_disp_image(item["image"], denorm=True)
        )
        for i in range(item["lm_heatmaps"].shape[0]):
            cv2.imwrite(
                temp_folder + "heatmap" + str(i) + ".jpg", item["lm_heatmaps"][i] * 255
            )
        show_landmark_heatmaps(
            item["lm_heatmaps"], filename=temp_folder + "heatmaps.jpg", nimgs=1
        )
        self.show_landmarks(
            cv2.resize(
                to_disp_image(item["image"], denorm=True),
                (lmcfg.HEATMAP_SIZE, lmcfg.HEATMAP_SIZE),
            ),
            item["landmarks"],
            filename=temp_folder + "overlapping.jpg",
        )

    def show_landmarks(self, img, landmarks, filename):
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 4, (255, 255, 0), -1)

        for x1, x2 in self.SKELETON:
            cv2.line(
                img,
                tuple(landmarks[x1 - 1]),
                tuple(landmarks[x2 - 1]),
                (255,),
                thickness=2,
            )
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

    def __len__(self):
        raise NotImplemented
