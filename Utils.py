import numpy as np
import matplotlib.pyplot as plt
import cv2


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
        img:       A numpy array image. It shape is (width, heigth, channels)
        keypoints: Key points about img parameter. It has total 24 key points and each key point has x,y positions.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    img = (img * 255).astype(dtype=np.uint8)
    keypoints = keypoints.astype(dtype=np.int)

    for keypoint in keypoints:
        cv2.circle(img, (keypoint[0], keypoint[1]), 4, (255, 0, 0), -1)
        
    plt.imshow(img)
    plt.axis("off")
    plt.show()
