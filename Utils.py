import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A


def train_val_split(imgs, keypoints, random_state):
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
    return imgs[trains], imgs[validations], keypoints[trains], keypoints[validations]


def visualize(img: np.ndarray, keypoints: np.ndarray):
    """
    Visualize the image and keypoints. Keypoints are represented on the image.
    @param:
        img:       A numpy array image. It shape is (height, width, channels)
        keypoints: Key points about img parameter. It has total 24 kinds of key points and each key point has x,y positions.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    # Convert img's data type to uint8. And img has (channels, height, width) shape, convert it to (height, width, channels) shape.
    img = (img * 255).astype(dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))

    # Convert key points data type to uint8, and if it has only 1 dimension, it means key points was flatten.
    # So convert to them to represent 2 dimension(key point label, 2(x, y)).
    keypoints = keypoints.astype(dtype=np.int)
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(-1, 2)

    # Draw keypoints on the image.
    for keypoint in keypoints:
        cv2.circle(img, (keypoint[0], keypoint[1]), 1, (255, 0, 0), -1)

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def preprocess(img: np.ndarray, keypoints: np.ndarray, model_name: str) -> tuple:
    """
    Pre-processing the image and keypoints for using as the model's input.
    @param
        img:       
        keypoints: 
        mode:      
    @return
        transformed_img:       
        transformed_keypoints: 
    """
    height, width, _ = img.shape
    square_len = min(height, width)

    if model_name == "resnet":
        transform = A.Compose(
            [
                A.CenterCrop(height=square_len, width=square_len, always_apply=True),
                A.Resize(height=224, width=224, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

        transformed = transform(image=img, keypoints=keypoints)

        transformed_img = transformed["image"]
        transformed_keypoints = np.array(transformed["keypoints"])

        return transformed_img, transformed_keypoints


def cvt_keypoint(keypoints: np.ndarray, heatmap_size: int or tuple) -> np.ndarray:
    """
    Convert keypoints to heatmaps. The converted heatmaps represents key points by gaussian distribution.
    @param
        keypoints:    A numpy array. It represents total 24 kinds of key points.
        heatmap_size: Converted heatmap's image size. It could be integer data type or tuple(height, width).
    @return

    """

    # TODO: define cvt_keypoint function.

    if isinstance(heatmap_size, int):
        heatmap_size = (heatmap_size, heatmap_size)

    heatmap = np.zeros(heatmap_size)
