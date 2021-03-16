import os

import pandas as pd
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from Utils import visualize, preprocess, train_val_split


class TrainDataSet(Dataset):
    def __init__(self, img_path: str, csv_path: str, transform):
        super(TrainDataSet, self).__init__()
        self.img_path = img_path  # path where images located.
        self.csv_path = csv_path  # path where csv file located.

        self.img_names, self.keypoints, self.class_labels = self._get_data(pd.read_csv(csv_path))

        self.transform = transform

    def __getitem__(self, idx: int) -> tuple:
        """
        Return items(image, keypoints, class_label) corresponding to the index parameter(idx).
        @param
            idx:                      Index corresponding to the data want to get.
        @return
            transformed_image:        The image tensor corresponding to the index.
            transformed_keypoints:    The coordinates tensor corresponding to the index.
        """
        # Get the image corresponding to the index.
        # Image loaded by cv2.imread has (height, width, channels) shape and BGR order.
        image = cv2.imread(os.path.join(self.img_path, self.img_names[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Get the keypoints corresponding to the index.
        keypoints = self.keypoints[idx]

        # Apply preprocessing transform to the image, keypoints.
        transformed = self.transform(image=image, keypoints=keypoints)
        transformed_image = transformed["image"]
        transformed_keypoints = transformed["keypoints"]

        # Convert image, coordinates to tensors suitable for use in the Pytorch model's input.
        # To use an image for pytorch, convert image to have (channels, height, width).
        transformed_image = torch.tensor(np.transpose(transformed_image, (2, 0, 1)), dtype=torch.float)
        transformed_keypoints = torch.flatten(torch.tensor(transformed_keypoints, dtype=torch.float))

        return transformed_image, transformed_keypoints

    def __len__(self):
        """
        Return the total number of dataset.
        @return
            data_size: the total number of dataset.
        """
        data_size = self.img_names.size
        return data_size

    def _get_data(self, dataframe: pd.DataFrame) -> tuple:
        """
        Get images' name list and Keypoints' coordinates.
        @param
            dataframe:     data frame from csv file. This has image name at index[0] and coordinates about keypoints at index[1:].
        @return
            img_names:     Numpy array that has all images name. E.g) "001-1-1-01-Z17_A-0000001.jpg"
            keypoints:     Numpy array that has all keypoints about each image. This shape is (data size, the num of label type, 2).
                           Shape's last dimension represents x, y position.
            column_labels: A string List of label types.
        """
        # The num of data.
        data_size = len(dataframe)

        # Get all images' names.
        img_names = dataframe.iloc[:, 0].to_numpy()

        # Get all coordinates about keypoints.
        keypoints = dataframe.iloc[:, 1:].to_numpy().reshape(data_size, -1, 2)

        # Get all label types.
        column_labels = [label.replace("_x", "").replace("_y", "") for label in dataframe.columns.tolist()[::2]]

        return img_names, keypoints, column_labels

    def divide_self(self):
        """
        Divide dataset to training dataset and validation dataset.
        @return
            train_dataset: The divided training data set.
            valid_dataset: The divided validation data set.
        """
        train_dataset = TrainDataSet(img_path=self.img_path, csv_path=self.csv_path, transform=self.transform)
        valid_dataset = TrainDataSet(img_path=self.img_path, csv_path=self.csv_path, transform=self.transform)

        train_imgs, valid_imgs, train_keypoints, valid_keypoints = train_val_split(
            imgs=self.img_names, keypoints=self.keypoints, random_state=42
        )

        train_dataset.img_names = train_imgs
        train_dataset.keypoints = train_keypoints

        valid_dataset.img_names = valid_imgs
        valid_dataset.keypoints = valid_keypoints

        return train_dataset, valid_dataset


class TestDataSet(Dataset):
    def __init__(self, img_path: str, csv_path: str, transform):
        super(TestDataSet, self).__init__()
        self.img_path = img_path
        self.csv_path = csv_path

        self.img_names = list(pd.read_csv(csv_path)["image"])
        self.class_labels = list(pd.read_csv(csv_path).columns[1:])

        self.transform = transform

    def __getitem__(self, idx: int):
        """
        Return an image corresponding to the index parameter(idx).
        @param
            idx:               Index corresponding to the data want to get.
        @return
            transformed_image: The image tensor corresponding to the index.
            original_shape:    The original image's shape.
        """
        # Get the image corresponding to the index.
        # Image loaded by cv2.imread has (height, width, channels) shape and BGR order.
        image = cv2.imread(os.path.join(self.img_path, self.img_names[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        original_shape = image.shape[:2]

        # Apply preprocessing transform to the image.
        transformed = self.transform(image=image)
        transformed_image = transformed["image"]
        transformed_image = torch.tensor(np.transpose(transformed_image, (2, 0, 1)), dtype=torch.float)

        file_name = self.img_names[idx]
        
        return transformed_image, original_shape, file_name

    def __len__(self):
        """
        Return the total number of dataset.
        @return
            data_size: the total number of dataset.
        """
        data_size = len(self.img_names)
        return data_size


if __name__ == "__main__":
    import albumentations as A
    import matplotlib.pyplot as plt

    # sanity check TrainDataSet Class.
    train_transform = A.Compose(
        [
            A.Resize(height=256, width=256, always_apply=True),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )
    train_dataset = TrainDataSet(img_path="./data/train_imgs", csv_path="./data/train_df.csv", transform=train_transform)
    train_img, keypoints = train_dataset[1]
    print(train_img.shape)
    print(keypoints.shape)
    print(train_dataset.class_labels)
    visualize(train_img, keypoints)

    # sanity check TestDataSet class.
    test_transform = A.Compose(
        [
            A.Resize(height=256, width=256, always_apply=True),
        ],
    )
    test_dataset = TestDataSet(img_path="./data/test_imgs", csv_path="./data/sample_submission.csv", transform=test_transform)
    test_img, origin_shape = test_dataset[1]
    print(test_img.shape)
    print(origin_shape)

    plt.imshow(test_img)
    plt.show()
