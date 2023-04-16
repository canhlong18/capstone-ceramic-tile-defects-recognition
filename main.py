"""Main program."""


from ultralytics import YOLO
from window import sliding_window as sw
import cv2


if __name__ == '__main__':
    # Load the high-resolution image
    image = cv2.imread("resources/images/_MG_2267.jpg", 1)

    # Load classification model
    model = YOLO(model="models/trained_set1_v5m6_best.pt")

    # Define the size of the sliding window and the stride
    window_size = (360, 360)
    stride = 240

    # Iterate through the sliding windows and process each window
    for x, y, window in sw.sliding_window(image, window_size, stride):
        # Perform object classification on the window
        results = model(source=window, stream=True, conf=0.3, iou=0.65, device=0)

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
                cv2.rectangle(image, (x_abs, y_abs), (x_abs + w_obj, y_abs + h_obj), (0, 255, 0), 2)

    # Display the high-resolution image with bounding boxes
    cv2.imshow('Predict with sliding window', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
