import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import neptune

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage


def train_val_split(imgs, keypoints, random_state=42):
    d = dict()
    for file in imgs:
        key = "".join(file.split("-")[:-1])
        if key not in d.keys():
            d[key] = [file]
        else:
            d[key].append(file)

    np.random.seed(random_state)
    trains = []
    validations = []
    for key, value in d.items():
        r = np.random.randint(len(value), size=2)
        for i in range(len(value)):
            if i in r:
                validations.append(np.where(imgs == value[i])[0][0])
            else:
                trains.append(np.where(imgs == value[i])[0][0])

    return (imgs[trains], imgs[validations], keypoints[trains], keypoints[validations])


def get_data_dicts(data_dir, imgs, keypoints):
    # train_dir = os.path.join(data_dir, "augmented" if phase=="train" else "train_imgs")
    train_dir = os.path.join(data_dir, "train_imgs")
    dataset_dicts = []

    for idx, item in tqdm(enumerate(zip(imgs, keypoints))):
        img, keypoint = item[0], item[1]

        record = {}
        filepath = os.path.join(train_dir, img)
        record["height"], record["width"] = cv2.imread(filepath).shape[:2]
        record["file_name"] = filepath
        record["image_id"] = idx

        keypoints_v = []
        for i, keypoint_ in enumerate(keypoint):
            keypoints_v.append(keypoint_)  # if coco set, should be added 0.5
            if i % 2 == 1:
                keypoints_v.append(2)

        x = keypoint[0::2]
        y = keypoint[1::2]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)

        obj = {"bbox": [x_min, y_min, x_max, y_max], "bbox_mode": BoxMode.XYXY_ABS, "category_id": 0, "keypoints": keypoints_v}

        record["annotations"] = [obj]
        dataset_dicts.append(record)

    return dataset_dicts


def draw_keypoints(image, keypoints, color=(0, 0, 255), diameter=5):
    keypoints_ = keypoints.copy()
    if len(keypoints_) == 48:
        keypoints_ = [[keypoints_[i], keypoints_[i + 1]] for i in range(0, len(keypoints_), 2)]

    assert isinstance(image, np.ndarray), "image argument does not numpy array."
    image_ = np.copy(image)
    for x, y in keypoints_:
        cv2.circle(image_, (int(x), int(y)), diameter, color, -1)

    return image_


def save_samples(dst_path, image_path, csv_path, mode="random", size=None, index=None):
    df = pd.read_csv(csv_path)

    if mode == "random":
        assert size is not None, "mode argument is random, but size argument is not given."
        choice_idx = np.random.choice(len(df), size=size, replace=False)
    if mode == "choice":
        assert index is not None, "mode argument is choice, but index argument is not given."
        choice_idx = index

    for idx in choice_idx:
        image_name = df.iloc[idx, 0]
        keypoints = df.iloc[idx, 1:]
        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)

        combined = draw_keypoints(image, keypoints)
        cv2.imwrite(os.path.join(dst_path, "sample" + image_name), combined)


class hook_neptune(HookBase):
    def __init__(self, max_iter):
        self._max_iter = max_iter

    def after_step(self):
        try:
            storage = get_event_storage()
            cur_iter = storage.iter

            if cur_iter == self._max_iter:
                # This hook only reports training progress (loss, ETA, etc) but not other data,
                # therefore do not write anything after training succeeds, even if this method
                # is called.
                return

            for k, v in storage.histories().items():
                if "loss" in k:
                    neptune.log_metric(k, cur_iter, v.median(20))
        except neptune.exceptions.NeptuneUninitializedException:
            pass
