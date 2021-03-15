import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from Models import create_model
from DataSets import KeyPointDataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(exp_name, model, criterion, optimizer, data_loader, num_epochs=25):
    print("===== training model starts =====")
    os.makedirs(f"./train_results/{exp_name}", exist_ok=True)
    train_loader, valid_loader = data_loader

    model.to(device)

    train_start_time = time.time()

    train_loss_history = []
    valid_loss_history = []

    for epoch_idx in range(1, num_epochs + 1):  # 1, ..., num_epochs
        epoch_start_time = time.time()

        iter_train_losses = []
        iter_valid_losses = []

        # Train Session.
        model.train()
        for iter_idx, (train_data, train_label) in enumerate(train_loader, 1):
            train_data, train_label = train_data.to(device), train_label.to(device)

            # Initialize optimizer.
            optimizer.zero_grad()

            train_pred = model(train_data)
            train_loss = criterion(train_pred, train_label)

            # Calculate parameters' gradients and Optimize the parameters.
            train_loss.backward()
            optimizer.step()

            iter_train_losses.append(train_loss.cpu().item())
            print(f"[epoch : {epoch_idx}/{num_epochs}] proceeding training iteration {iter_idx}", end="\r", flush=True)

        # Validation session.
        model.eval()
        with torch.no_grad():
            for iter_idx, (valid_data, valid_label) in enumerate(valid_loader, 1):
                valid_data, valid_label = valid_data.to(device), valid_label.to(device)

                valid_pred = model(valid_data)
                valid_loss = criterion(valid_pred, valid_label)

                iter_valid_losses.append(valid_loss.cpu().item())
                print(f"[epoch : {epoch_idx}/{num_epochs}] proceeding validation iteration {iter_idx}", end="\r", flush=True)

        # Calculate train loss and validation loss.
        train_loss = np.mean(iter_train_losses)
        valid_loss = np.mean(iter_valid_losses)

        # Record train loss, validation loss history.
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        time_per_epoch = time.time() - epoch_start_time

        print(
            f"[epoch : {epoch_idx}/{num_epochs}] train loss : {train_loss:0.4f} | validation loss : {valid_loss:0.4f} | time taken : {time_per_epoch:0.2f}",
            end="",
        )
        if valid_loss <= min(valid_loss_history):
            print(" -> minumum validation loss. model weight saved.")
            torch.save(model.state_dict(), f"./train_results/{exp_name}/model_weight.pth")
        else:
            print()

    train_end_time = time.time() - train_start_time
    print("===== training model is all done. =====")
    print(f"total training taken time : {train_end_time}")


# TODO: Apply neptune API and record training logs on neptune experiments.
if __name__ == "__main__":
    # Hyper Parameters.
    experiment_name = "coordiantes regression with resnet"
    epoch_nums = 25
    batch_size = 64
    lr = 1e-3
    model_name = "resnet50"

    model = create_model(class_num=48, model_name=model_name)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    dataset = KeyPointDataSet(img_path="./data/train_imgs", csv_path="./data/train_df.csv", model_type="resnet")
    train_dataset, valid_dataset = dataset.divide_self()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    train(
        exp_name=experiment_name,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=[train_loader, valid_loader],
    )
