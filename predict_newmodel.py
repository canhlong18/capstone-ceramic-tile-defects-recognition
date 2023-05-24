"""This module perform predicting new trained-model."""

from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    freeze_support()

    # use trained model to predict
    model = YOLO(model="models/based-models/best.pt")
    model.info(verbose=True)

    # read image
    img_path = r"datasets/ceramic-tile-defects.v1i/test/images"
    # image = cv2.imread(img_path, 1)
    # img = cv2.resize(img, (720, 720))

    # predict on image and save results
    results = model(source=img_path,
                    conf=0.15, iou=0.65,
                    device=0, show=False,
                    save=True, save_crop=False, save_txt=True, save_conf=True)

    # print(results)
    # print(results.xyxy[0])

    # export model to other formats
    # model.fuse()
    # model.export(format="onnx")
