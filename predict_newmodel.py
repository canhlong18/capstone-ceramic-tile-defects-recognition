"""This module perform predicting new trained-model."""


from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2


if __name__ == "__main__":
    freeze_support()

    # use trained model to predict
    model = YOLO(model="models/set2-v5m6-SmallObject.pt")
    model.info(verbose=True)

    # read image
    img_path = r"C:\Projects\PyCharm\tile-errors\datasets\ceramic-tile-defects2-smallobects_v9\test\images"
    image = cv2.imread(img_path, 1)
    # img = cv2.imread("resources/images", 1)
    # img = cv2.resize(img, (720, 720))

    # predict on image and save results
    result = model(source=img_path,
                   conf=0.25, iou=0.65,
                   device=0, show=False,
                   save=True, save_crop=True)

    # export model to other formats
    # model.fuse()
    # model.export(format="onnx")
