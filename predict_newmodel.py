"""This module perform predicting new trained-model."""


from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2


if __name__ == "__main__":
    freeze_support()

    # use trained model to predict
    model = YOLO(model="models/trained_set1_v5m6_best.pt")
    model.info(verbose=True)

    # read image
    img = cv2.imread("resources/images", 1)
    img = cv2.resize(img, (720, 720))

    # predict on image and save results
    result = model(source="resources/images",
                   conf=0.25, iou=0.65,
                   device=0, show=False,
                   save=True, save_crop=True)

    # export model to other formats
    model.fuse()
    model.export(format="onnx")
