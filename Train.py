import time
import os

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from Models import create_model
from DataSets import TrainDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configs.
EXPERIMENT_TYPE = "coordinates regression"
MODEL_TYPE = "resnet"
EPOCH_NUM = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DATA_PATH = "./data"

# define model/criterion/optimizer
model = create_model(class_num=48, model_name=MODEL_TYPE)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# define transform for augmentation.
transform = A.Compose(
    [
        A.Resize(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ],
    keypoint_params=A.KeypointParams(format="xy"),
)

# define dataset/dataloader.
dataset = TrainDataSet(
    img_path=os.path.join(DATA_PATH, "/pre-processed/cropped2_train_imgs"), csv_path=os.path.join(DATA_PATH, "pre-processed/cropped2_train_df.csv"), transform=transform
)
train_dataset, valid_dataset = dataset.divide_self()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# train session.
print("===== training model starts =====")
os.makedirs(f"./train_results/{EXPERIMENT_TYPE}/{MODEL_TYPE}", exist_ok=True)

model.to(device)

train_start_time = time.time()

train_loss_history = []
valid_loss_history = []

for epoch_idx in range(1, EPOCH_NUM + 1):  # 1, ..., num_epochs
    epoch_start_time = time.time()

    iter_train_losses = []
    iter_valid_losses = []

    model.train()
    for iter_idx, (train_data, train_label) in enumerate(train_loader, 1):
        train_data, train_label = train_data.to(device), train_label.to(device)

        # initialize optimizer.
        optimizer.zero_grad()

        train_pred = model(train_data)
        train_loss = criterion(train_pred, train_label)

        # calculate parameters' gradients and Optimize the parameters.
        train_loss.backward()
        optimizer.step()

        iter_train_losses.append(train_loss.cpu().item())
        print(f"[epoch : {epoch_idx}/{EPOCH_NUM}] proceeding training iteration {iter_idx}", end="\r", flush=True)

    # validation session.
    model.eval()
    with torch.no_grad():
        for iter_idx, (valid_data, valid_label) in enumerate(valid_loader, 1):
            valid_data, valid_label = valid_data.to(device), valid_label.to(device)

            valid_pred = model(valid_data)
            valid_loss = criterion(valid_pred, valid_label)

            iter_valid_losses.append(valid_loss.cpu().item())
            print(f"[epoch : {epoch_idx}/{EPOCH_NUM}] proceeding validation iteration {iter_idx}", end="\r", flush=True)

    # calculate train loss and validation loss.
    train_loss = np.mean(iter_train_losses)
    valid_loss = np.mean(iter_valid_losses)

    # record train loss, validation loss history.
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    time_per_epoch = time.time() - epoch_start_time

    print(
        f"[epoch : {epoch_idx}/{EPOCH_NUM}] train loss : {train_loss:0.4f} | validation loss : {valid_loss:0.4f} | time taken : {time_per_epoch:0.2f}",
        end="",
    )
    if valid_loss <= min(valid_loss_history):
        print(" -> minumum validation loss. model weight saved.")
        torch.save(model.state_dict(), f"./train_results/{EXPERIMENT_TYPE}/{MODEL_TYPE}/model_weight.pth")
    else:
        print()

train_end_time = time.time() - train_start_time
print("===== training model is all done. =====")
print(f"total training taken time : {train_end_time}")
