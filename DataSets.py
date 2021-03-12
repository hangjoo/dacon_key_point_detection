import os

import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset


class KeyPointDataSet(Dataset):
    def __init__(self, img_path: str, csv_path: str):
        super(Dataset, self).__init__()
        self.img_path = img_path  # path where images located.
        self.csv_path = csv_path  # path where csv file located.

        self.img_names, self.coordinates, self.class_labels = self._get_data(pd.read_csv(csv_path))

    def __getitem__(self, idx: int) -> tuple:
        """
        Return items(image, coordinates, class_label) corresponding to the index parameter(idx).
        @param
            idx: Index corresponding to the data want to get.
        @return
            img:          the image tensor corresponding to the index.
            coordinates:  the coordinates tensor corresponding to the index.
            class_labels: class labels list.
        """
        # get the image corresponding to the index.
        img = cv2.imread(os.path.join(self.img_path, self.img_names[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img) / 255.

        # get the coordinates corresponding to the index.
        coordinates = torch.tensor(self.coordinates[idx])

        return img, coordinates

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
            dataframe: data frame from csv file. This has image name at index[0] and coordinates about keypoints at index[1:].
        @return
            img_names:     Numpy array that has all images name. E.g) "001-1-1-01-Z17_A-0000001.jpg"
            coordinates:   Numpy array that has all coordinates about each image. This shape is (data size, the num of label type, 2).
                           Shape's last dimension represents x, y position.
            column_labels: A string List of label types.
        """
        # the num of data.
        data_size = len(dataframe)

        # get all images' names.
        img_names = dataframe.iloc[:, 0].to_numpy()

        # get all coordinates about keypoints.
        coordinates = dataframe.iloc[:, 1:].to_numpy().reshape(data_size, -1, 2)

        # get all label types.
        column_labels = [label.replace("_x", "").replace("_y", "") for label in dataframe.columns.tolist()[::2]]

        return img_names, coordinates, column_labels


if __name__ == "__main__":
    test_dataset = KeyPointDataSet(img_path="data/train_imgs", csv_path="data/train_df.csv")
    img, coordinates = test_dataset[0]
    print(img)
    print(coordinates.shape)
    print(test_dataset.class_labels)
