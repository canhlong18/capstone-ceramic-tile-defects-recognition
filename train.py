"""This module perform training model with datasets from roboflow."""


from roboflow import Roboflow
from ultralytics import YOLO
import os
import config


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DATASET_DIRECTORY"] = f"{ROOT_DIR}\datasets"

rf_apikey = config.roboflow.get('api_key')
rf_workspace = config.roboflow.get('workspace')
rf_project = config.roboflow.get('project')
rf_version = config.roboflow.get('version')


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
    model.train(data="datasets/Car-License-Plate-2/data.yaml",
                epochs=10, batch=8, imgsz=640, patience=50,
                cache=False, device=0, workers=8)

    # validate model and check model info
    model.val()
    model.info(verbose=True)
