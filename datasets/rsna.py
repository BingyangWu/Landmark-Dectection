import os
import time
import numpy as np
import torch.utils.data as td
import pandas as pd

from utils import log, geometry, nn
from utils import vis
from datasets import handdataset
import config as cfg
import json
from cv2 import imwrite


class RSNA(handdataset.HandDataset):
    def __init__(
        self,
        root,
        cache_root=None,
        train=True,
        return_modified_images=False,
        **kwargs,
    ):

        self.split_folder = "train" if train else "val"
        self.split_folder = os.path.join(root, self.split_folder, "images")
        self.roi_folder = os.path.join(
            root, "keypoints/RSNA_ANNOTATIONS/ANATOMICAL_ROIS"
        )
        self.json_filename = (
            "RSNA_Anatomical_ROIs_Training.json"
            if train
            else "RSNA_Anatomical_ROIs_Validation.json"
        )
        self.json_filename = os.path.join(self.roi_folder, self.json_filename)
        fullsize_img_dir = os.path.join(root, self.split_folder)
        self.annotation_filename = "loose_bb_{}.csv".format(self.split_folder)
        self.bad_point = 0

        super().__init__(
            root=root,
            cache_root=cache_root,
            fullsize_img_dir=fullsize_img_dir,
            crop_dir=os.path.join(self.split_folder, "crops"),
            # return_landmark_heatmaps=True,
            return_modified_images=return_modified_images,
            **kwargs,
        )

        # shuffle images since dataset is sorted by identities
        import sklearn.utils

        self.annotations = sklearn.utils.shuffle(self.annotations)
        print("removed {} bad datapoints from RSNA".format(self.bad_point))
        print("source:", self.split_folder)

    @property
    def cropped_img_dir(self):
        return os.path.join(self.cache_root, self.split_folder, "crops")

    def get_crop_extend_factors(self):
        return 0.05, 0.1

    @property
    def ann_csv_file(self):
        return os.path.join(self.root, self.meta_folder, self.annotation_filename)

    def _read_annots_from_csv(self):
        print("Reading CSV file...")
        annotations = pd.read_csv(self.ann_csv_file)
        print(f"{len(annotations)} lines read.")

        # assign new continuous ids to persons (0, range(n))
        print("Creating id labels...")
        _ids = annotations.NAME_ID
        _ids = _ids.map(lambda x: int(x.split("/")[0][1:]))
        annotations["ID"] = _ids

        return annotations

    def _load_annotations(self, split):
        def toint(x):
            y = []
            for each in x:
                y.append(int(each))
            return y

        data = {"id": [], "filename": [], "W": [], "H": [], "bbox": [], "landmarks": []}
        with open(self.json_filename) as f:
            annots = json.load(f)
        for each, other in zip(annots["images"], annots["annotations"]):
            each.update(other)
            if each["bbox"][2] == 0:
                self.bad_point += 1
                continue
            data["id"].append(each["id"])
            data["filename"].append(os.path.join(self.split_folder, each["file_name"]))
            data["W"].append(each["width"])
            data["H"].append(each["height"])
            data["bbox"].append(toint(each["bbox"]))
            lms = each["keypoints"]
            formatted_lms = []
            for i in range(self.NUM_LANDMARKS):
                formatted_lms.append((lms[i * 3], lms[i * 3 + 1]))
            formatted_lms = np.array(formatted_lms)
            data["landmarks"].append(np.array(formatted_lms))
        return pd.DataFrame(data)

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.H.values

    @property
    def widths(self):
        return self.annotations.W.values

    @staticmethod
    def _get_identity(sample):
        return sample.ID

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        def XYWH2XYXY(bb):
            return [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]

        sample = self.annotations.iloc[idx]
        # bb = self.get_adjusted_bounding_box(sample.X, sample.Y, sample.W, sample.H)
        bb = XYWH2XYXY(sample.bbox)
        landmarks_for_crop = sample.landmarks.astype(np.float32)
        return self.get_sample(sample.filename, bb, landmarks_for_crop)


cfg.register_dataset(RSNA)

if __name__ == "__main__":
    # extract_main()
    # exit()
    from utils.nn import Batch
    from utils import random

    random.init_random()

    ds = VggFace2(
        train=True,
        deterministic=True,
        use_cache=False,
        align_face_orientation=False,
        return_modified_images=False,
        image_size=256,
    )
    micro_batch_loader = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    f = 1.0
    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print("t load:", time.perf_counter() - t)
        t = time.perf_counter()
        batch = Batch(data, gpu=False)
        print("t Batch:", time.perf_counter() - t)
        images = nn.denormalized(batch.images)
        vis.vis_square(images, fx=f, fy=f, normalize=False, nCols=10, wait=0)
