import os

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from utils import train_val_split, get_data_dicts, hook_neptune, save_samples
from Trainer import Trainer

import neptune
import neptune_config


def main():
    data_name = "augmented_2"
    data_path = os.path.join("./data", data_name)
    csv_name = data_name + ".csv"

    train_df = pd.read_csv(os.path.join(data_path, csv_name))

    keypoint_names = list(map(lambda x: x[:-2], train_df.columns.to_list()[1::2]))
    keypoint_flip_map = [
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_palm", "right_palm"),
        ("left_instep", "right_instep"),
    ]

    image_list = train_df.iloc[:, 0].to_numpy()
    keypoints_list = train_df.iloc[:, 1:].to_numpy()
    train_imgs, valid_imgs, train_keypoints, valid_keypoints = train_val_split(image_list, keypoints_list, random_state=42)

    image_set = {"train": train_imgs, "valid": valid_imgs}
    keypoints_set = {"train": train_keypoints, "valid": valid_keypoints}

    hyper_params = {
        "augmented_ver": data_name,
        "learning_rate": 0.001,
        "num_epochs": 10000,
        "batch_size": 256,
        "description": "Final training"
    }

    ns = neptune.init(project_qualified_name="hangjoo/Dacon-motion-keypoint-detection", api_token=neptune_config.token)
    neptune.create_experiment(name="detectron2", params=hyper_params, upload_source_files="./code/train.py")
    experiment_id = ns._get_current_experiment()._id

    for phase in ["train", "valid"]:
        DatasetCatalog.register(
            "keypoints_" + phase, lambda phase=phase: get_data_dicts(data_path, image_set[phase], keypoints_set[phase])
        )
        MetadataCatalog.get("keypoints_" + phase).set(thing_classes=["human"])
        MetadataCatalog.get("keypoints_" + phase).set(keypoint_names=keypoint_names)
        MetadataCatalog.get("keypoints_" + phase).set(keypoint_flip_map=keypoint_flip_map)
        MetadataCatalog.get("keypoints_" + phase).set(evaluator_type="coco")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("keypoints_train",)
    cfg.DATASETS.TEST = ("keypoints_valid",)
    cfg.DATALOADER.NUM_WORKERS = 16  # On Windows environment, this value must be 0.
    cfg.SOLVER.IMS_PER_BATCH = 2  # mini batch size would be (SOLVER.IMS_PER_BATCH) * (ROI_HEADS.BATCH_SIZE_PER_IMAGE).
    cfg.SOLVER.BASE_LR = hyper_params["learning_rate"]  # Learning Rate.
    cfg.SOLVER.MAX_ITER = hyper_params["num_epochs"]  # Max iteration.
    cfg.SOLVER.GAMMA = 0.8
    cfg.SOLVER.STEPS = [3000, 4000, 5000, 6000, 7000, 8000]  # The iteration number to decrease learning rate by GAMMA.
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = hyper_params["batch_size"]  # Use to calculate RPN loss.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 24
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((24, 1), dtype=float).tolist()
    cfg.TEST.EVAL_PERIOD = 5000  # Evaluation would occur for every cfg.TEST.EVAL_PERIOD value.
    cfg.OUTPUT_DIR = os.path.join("./output", experiment_id)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.register_hooks([hook_neptune(max_iter=hyper_params["num_epochs"])])
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    test_dir = os.path.join("data", "test_imgs")
    test_list = os.listdir(test_dir)
    test_list.sort()
    except_list = []

    files = []
    preds = []
    for file in tqdm(test_list):
        filepath = os.path.join(test_dir, file)
        # print(filepath)
        im = cv2.imread(filepath)
        outputs = predictor(im)
        outputs = outputs["instances"].to("cpu").get("pred_keypoints").numpy()
        files.append(file)
        pred = []
        try:
            for out in outputs[0]:
                pred.extend([float(e) for e in out[:2]])
        except IndexError:
            pred.extend([0] * 48)
            except_list.append(filepath)
        preds.append(pred)

    df_sub = pd.read_csv("./data/sample_submission.csv")
    df = pd.DataFrame(columns=df_sub.columns)
    df["image"] = files
    df.iloc[:, 1:] = preds

    df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{experiment_id}_submission.csv"), index=False)
    if except_list:
        print(
            "The following images are not detected keypoints. The row corresponding that images names would be filled with 0 value."
        )
        print(*except_list)
    save_samples(cfg.OUTPUT_DIR, test_dir, os.path.join(cfg.OUTPUT_DIR, f"{experiment_id}_submission.csv"), mode="random", size=5)
    neptune.stop()


if __name__ == "__main__":
    main()
