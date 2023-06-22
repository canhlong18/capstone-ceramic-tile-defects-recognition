"""This module perform training model with datasets from roboflow."""

from roboflow import Roboflow
from ultralytics import YOLO
import os
import configs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DATASET_DIRECTORY"] = f"{ROOT_DIR}\\datasets"

rf_apikey = configs.roboflow.get('api_key')
rf_workspace = configs.roboflow.get('workspace')
rf_project = configs.roboflow.get('project')
rf_version = configs.roboflow.get('version')

if __name__ == '__main__':
    print("CAPSTONE PROJECT: training model process.")
    print("DATASET_DIRECTORY in:", os.environ.get("DATASET_DIRECTORY"))

    # get dataset from roboflow
    rf = Roboflow(api_key=rf_apikey)
    project = rf.workspace(rf_workspace).project(rf_project)
    dataset = project.version(rf_version).download("yolov5")

    # train model
    model = YOLO(model="yolov5m6u.yaml")

    # cache =False(disk), =True(ram);
    # device =None, =0,1,2,3(cuda gpu device), =cpu
    model.train(data="datasets/ceramic-tile-defects.v1i/data.yaml",
                epochs=100, batch=8, imgsz=1280, patience=50,
                cache=False, device=0, workers=8)

    # validate model and check model info
    model.val()
    model.info(verbose=True)
