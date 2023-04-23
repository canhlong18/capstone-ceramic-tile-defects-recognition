from ultralytics import YOLO
from window import sliding_window as sw
from pathlib import Path
import cv2


model = YOLO(model="models/set2-v5m6-SmallObject.pt")

# Load the high-resolution image
img_path = "resources/images/_MG_2286.jpg"
image = cv2.imread(img_path, 1)
image = cv2.resize(image, (1280, 1280))

show_image = cv2.imread(img_path, 1)
show_image = cv2.resize(show_image, (1280, 1280))

# Define the size of the sliding window and the stride
window_size = (312, 312)
stride = 240

# Other variables saving prediction results
current_window = 0
first_execute = 1
save_dir = ""


# Iterate through the sliding windows and process each window
for x, y, window in sw.sliding_window(image, window_size, stride):

    # Perform object detection on the window
    results = model(source=window,
                    conf=0.25, iou=0.65,
                    device=0, save=True, save_crop=True, show=True)
    # results = model(source=window, stream=True, conf=0.3, iou=0.65, device=0)

    # find save-directory
    if first_execute == 1:
        save_dir = str([p for p in Path("./runs/detect").iterdir() if p.is_dir()][-1])
        print(save_dir)
        first_execute = 0

    # rename new predicted window image in order to avoid overriding on old images
    predicted_img = Path(f"{save_dir}/image0.jpg")
    predicted_img.rename(f"{save_dir}/window{str(current_window)}.jpg")
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
            x_abs, y_abs = x + x_obj, y + y_obj

            # Draw bounding box on original image
            cv2.rectangle(show_image, (x_abs, y_abs), (x_abs + w_obj, y_abs + h_obj), (0, 255, 0), 2)

# Display the high-resolution image with bounding boxes
window_name = "Prediction with sliding window"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 720)

cv2.imshow(window_name, show_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
