import os
import numpy as np
from tqdm import tqdm
import cv2
from detectron2.structures import BoxMode


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
