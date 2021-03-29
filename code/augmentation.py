import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import albumentations as A

keypoint_params = A.KeypointParams(format="xy", label_fields=["class_labels"], remove_invisible=False, angle_in_degrees=True)
transform_dict = {
    "Original": A.Compose([A.RandomCrop(height=1080, width=1920, p=1)], keypoint_params=keypoint_params),
    "CenterCrop": A.Compose([A.CenterCrop(height=720, width=1280, p=1)], keypoint_params=keypoint_params),
    "RandomCrop_1": A.Compose([A.RandomCrop(height=720, width=960, p=1)], keypoint_params=keypoint_params),
    "RandomCrop_2": A.Compose([A.RandomCrop(height=540, width=720, p=1)], keypoint_params=keypoint_params),
    "RandomCrop_3": A.Compose([A.RandomCrop(height=960, width=960, p=1)], keypoint_params=keypoint_params),
    "Rotate45": A.Compose([A.Rotate(limit=45, p=1)], keypoint_params=keypoint_params),
    "Rotate45_CenterCrop": A.Compose([A.Rotate(limit=45, p=1), A.CenterCrop(height=720, width=1280, p=1)], keypoint_params=keypoint_params),
    "Rotate45_RandomCrop_1": A.Compose([A.Rotate(limit=45, p=1), A.RandomCrop(height=720, width=960, p=1)], keypoint_params=keypoint_params),
    "Rotate45_RandomCrop_2": A.Compose([A.Rotate(limit=45, p=1), A.RandomCrop(height=540, width=720, p=1)], keypoint_params=keypoint_params),
    "Rotate45_RandomCrop_3": A.Compose([A.Rotate(limit=45, p=1), A.RandomCrop(height=960, width=960, p=1)], keypoint_params=keypoint_params),
}


def main():
    data_path = "./data"
    src_path = os.path.join(data_path, "original")
    src_image_path = os.path.join(src_path, "train_imgs")
    src_df = pd.read_csv(os.path.join(src_path, "train_df_modified.csv"))

    keypoints_labels = list(map(lambda x: x[:-2], src_df.columns[1:].tolist()[::2]))
    image_list = src_df.iloc[:, 0].to_numpy()
    keypoints_list = src_df.iloc[:, 1:].to_numpy()
    paired_keypoints_list = keypoints_list.reshape(-1, 24, 2)

    dst_name = "augmented_1"
    dst_path = os.path.join(data_path, dst_name)
    dst_image_path = os.path.join(dst_path, "train_imgs")

    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(dst_image_path, exist_ok=True)

    augmented_image_list = []
    augmented_keypoints_list = []
    for image_name, paired_keypoints in tqdm(zip(image_list, paired_keypoints_list)):
        src_image = cv2.imread(os.path.join(src_image_path, image_name))

        transform_names = [
            "Original",
            "RandomCrop_1",
            "RandomCrop_2",
            "CenterCrop",
            "Rotate45",
            "Rotate45_RandomCrop_1",
            "Rotate45_RandomCrop_2",
            "Rotate45_CenterCrop",
        ]

        for transform_name in transform_names:
            augmented = transform_dict[transform_name](image=src_image, keypoints=paired_keypoints, class_labels=keypoints_labels)
            augmented_image = augmented["image"]
            augmented_keypoints = np.array(augmented["keypoints"]).flatten()
            augmented_name = f"{transform_name}_{image_name}"

            cv2.imwrite(os.path.join(dst_image_path, augmented_name), augmented_image)
            augmented_image_list.append(augmented_name)
            augmented_keypoints_list.append(augmented_keypoints)

    dst_df = pd.DataFrame(columns=src_df.columns)
    dst_df["image"] = augmented_image_list
    dst_df.iloc[:, 1:] = augmented_keypoints_list
    dst_df.to_csv(os.path.join(dst_path, dst_name + ".csv"), index=False)


if __name__ == "__main__":
    main()
