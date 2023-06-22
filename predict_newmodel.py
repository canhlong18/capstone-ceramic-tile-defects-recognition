"""This module perform predicting new trained-model."""

from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    freeze_support()

    # use trained model to predict
    model = YOLO(model="models/based-models/best.pt")
    model.info(verbose=True)

    # export model to other formats
    # model.fuse()
    # model.to(device='CUDA')
    # model.export(format='engine')

    # read image
    vid_path = f"resources/videos/ceramic-tile-defects.mp4"
    img_path = r"datasets/ceramic-tile-defects.v1i/test/images"
    image = cv2.imread(img_path, 1)
    image = cv2.resize(image, (1280, 1280))

    # predict on image and save results
    results = model(source=vid_path, vid_stride=30,
                    conf=0.2, iou=0.7,
                    device=0, show=False, visualize=False,
                    save=True, save_crop=False, save_txt=True, save_conf=True)

    # Streaming video
    cap_vid = cv2.VideoCapture(vid_path)

    window_name = "Prediction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 720)

    while cap_vid.isOpened():
        success, frame = cap_vid.read()

        if success:
            results = model(source=frame)
            draw_frame = results[0].plot()
            cv2.imshow(window_name, draw_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap_vid.release()
    cv2.destroyAllWindows()
