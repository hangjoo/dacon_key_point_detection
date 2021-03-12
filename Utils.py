import numpy as np


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
