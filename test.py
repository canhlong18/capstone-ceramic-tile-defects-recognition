from window import sliding_window as sw
import cv2


if __name__ == '__main__':

    image = cv2.imread("resources/images/_MG_2264.jpg", 1)

    window_size = (360, 360)
    stride = 240

    for x, y, window in sw.sliding_window(image, window_size, stride):
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 720, 720)

        cv2.imshow('img', window)
        cv2.waitKey(0)

    print("image size: ", image.shape[:2])

    cv2.destroyAllWindows()
