import gdown
import os
from colorama import Fore
import numpy as np
import torch

# Models
model_weights = {"yolov8s.pt": "https://drive.google.com/file/d/17z52g4E0hQ1VKVCNRxj3mzuay9oXyNZ5/view?usp=drive_link"}


def download_weights_from_drive(url, output_path):
    gdown.download_folder(url, quiet=False, output=output_path, use_cookies=False)


def download_model(model_path, model_name):
    model_directory = os.path.join(model_path, model_name)
    if not os.path.exists(model_directory):
        print(Fore.RED + 'Model Not Found\nDownloading the model' + Fore.RESET)
        # converting link
        link = model_weights[model_name]
        file_id = link.split('/')[-2]
        prefix = "https://drive.google.com/uc?/exports=download&id="
        link = prefix+file_id
        #

        download_weights_from_drive(url=link, output_path=model_directory)


def get_indexes(model_classes, expected_classes):
    """*************************************
    Functionality: find the indexes of exp_classes(list of string) form model_classes(list of string)
    Parameters: model_classes(list of yolo model class), exp_classes(list of our expected classes)
    Returns: List of indexes
    ****************************************"""
    if isinstance(model_classes, dict):
        model_classes = list(model_classes.values())

    indexes = []
    for i, label in enumerate(expected_classes):
        if label in model_classes:
            indexes.append(model_classes.index(label))

    return indexes


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height

    return y
