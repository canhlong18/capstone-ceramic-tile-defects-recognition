import cv2
import math
from pathlib import Path
from ultralytics import YOLO
from window.sliding_window import sliding_window


def get_latest_save_dir():
    save_dir_list = [item for item in Path("runs/detect").iterdir() if item.is_dir()]
    if save_dir_list:
        sorted_save_dir = sorted(save_dir_list, key=lambda folder: folder.stat().st_ctime)
        return str(sorted_save_dir[-1])
    else:
        return


model = YOLO(model="models/tiling-models/best.pt")

# Load the high-resolution image
img_path = Path("resources/for-demo/_MG_2481_.jpg")
image = cv2.imread(str(img_path), 1)
show_image = cv2.imread(str(img_path), 1)

# Define the size of the sliding window and the stride
window_size = (312, 312)
stride = 240

# Other variables saving prediction results
current_window = 0
save_dir = ""


# Iterate through the sliding windows and process each window
for x, y, window in sliding_window(image, window_size, stride):
    # Perform object detection on the window
    results = model(source=window, show=False,
                    device=0, conf=0.15, iou=0.65,
                    save=True, save_crop=True, save_txt=True, save_conf=True)

    # find save-directory
    save_dir = get_latest_save_dir()

    # rename new predicted window image in order to avoid overriding on old images
    Path(f"{save_dir}/image0.jpg").rename(f"{save_dir}/window{str(current_window)}.jpg")
    current_window += 1

    # Process the detected objects or take other actions
    for r in results:
        names = r.names
        boxes = r.boxes

        for box in boxes:
            # Extract object coordinates
            x_obj, y_obj, w_obj, h_obj = box.xywh[0]
            x_obj, y_obj, w_obj, h_obj = int(x_obj), int(y_obj), int(w_obj), int(h_obj)

            # Compute absolute coordinates on the high-resolution image
            x_abs, y_abs = x + x_obj - round(w_obj / 2), y + y_obj - round(h_obj / 2)

            # class name and confidence
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = names[int(box.cls[0])]

            # Draw bounding box on original image
            cv2.rectangle(img=show_image,
                          pt1=(x_abs, y_abs), pt2=(x_abs + w_obj, y_abs + h_obj),
                          color=(0, 255, 0), thickness=3)

            cv2.putText(img=show_image, text=f"{cls} - {conf}",
                        org=(max(0, x_abs), max(20, y_abs - 5)),
                        fontFace=cv2.FONT_ITALIC, fontScale=3,
                        color=(133, 83, 245), thickness=4)

cv2.imwrite(f"{save_dir}/{img_path.name}", show_image)
Path(f"{save_dir}/labels/image0.txt").rename(f"{save_dir}/labels/{img_path.stem}.txt")

# Display the high-resolution image with bounding boxes
window_name = "Prediction with sliding window"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 720)
cv2.imshow(window_name, show_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
