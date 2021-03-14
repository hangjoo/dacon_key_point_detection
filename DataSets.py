import os

import pandas as pd
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from Utils import visualize, preprocess


class KeyPointDataSet(Dataset):
    def __init__(self, img_path: str, csv_path: str):
        super(Dataset, self).__init__()
        self.img_path = img_path  # path where images located.
        self.csv_path = csv_path  # path where csv file located.

        self.img_names, self.keypoints, self.class_labels = self._get_data(pd.read_csv(csv_path))

    def __getitem__(self, idx: int) -> tuple:
        """
        Return items(image, keypoints, class_label) corresponding to the index parameter(idx).
        @param
            idx:          Index corresponding to the data want to get.
        @return
            img:          The image tensor corresponding to the index.
            keypoints:  The coordinates tensor corresponding to the index.
            class_labels: Class labels list.
        """
        # Get the image corresponding to the index.
        # Image loaded by cv2.imread has (height, width, channels) shape and BGR order.
        img = cv2.imread(os.path.join(self.img_path, self.img_names[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # Get the keypoints corresponding to the index.
        keypoints = self.keypoints[idx]

        # Convert image, coordinates to tensors suitable for use in the Pytorch model's input.
        # To use an image for pytorch, convert image to have (channels, height, width).
        img, keypoints = preprocess(img=img, keypoints=keypoints, mode="resnet")
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        keypoints = torch.from_numpy(keypoints)

        return img, keypoints

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
            keypoints:   Numpy array that has all keypoints about each image. This shape is (data size, the num of label type, 2).
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


if __name__ == "__main__":
    test_dataset = KeyPointDataSet(img_path="data/train_imgs", csv_path="data/train_df.csv")
    img, keypoints = test_dataset[1]

    print(img.shape)
    print(keypoints.shape)
    print(test_dataset.class_labels)

    visualize(img, keypoints)
    print(torch.min(img))
    print(torch.max(img))
