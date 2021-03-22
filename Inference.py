import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Models import create_model
from DataSets import TestDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_TYPE = "coordinates regression"
MODEL_TYPE = "resnet"
INPUT_SIZE = 224
DATA_PATH = "./data"
MODEL_PATH = "./train_results"

# define model.
model = create_model(class_num=48, model_name=MODEL_TYPE).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, EXPERIMENT_TYPE, MODEL_TYPE, "model_weight.pth")))

# define transform.
transform = A.Compose(
    [
        A.Resize(height=INPUT_SIZE, width=INPUT_SIZE, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ],
)

# define dataset/dataloader.
test_dataset = TestDataSet(img_path=os.path.join(DATA_PATH, "test_imgs"), csv_path=os.path.join(DATA_PATH, "sample_submission.csv"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# inference session.
original_height = []
original_width = []
inference_results = []
for test_idx, (inference_image, original_shape, file_name) in enumerate(test_loader):
    inference_image = inference_image.to(device)

    inference_pred = model(inference_image)

    inference_results.extend(inference_pred.cpu().tolist())
    original_height.extend(original_shape[0].tolist())
    original_width.extend(original_shape[1].tolist())
    
    print(f"[{test_idx + 1}/{len(test_loader)}]")

original_height = np.array(original_height)
original_width = np.array(original_width)
inference_results = np.array(inference_results)

for i in range(inference_results.shape[0]):
    for j in range(inference_results.shape[1]):
        if j % 2 == 0:
            inference_results[i][j] = inference_results[i][j] * original_width[i] / INPUT_SIZE
        else:
            inference_results[i][j] = inference_results[i][j] * original_height[i] / INPUT_SIZE

df_sub = pd.read_csv("./data/sample_submission.csv")
df = pd.DataFrame(columns=df_sub.columns)
df['image'] = test_dataset.img_names
df.iloc[:, 1:] = inference_results
df.head()
