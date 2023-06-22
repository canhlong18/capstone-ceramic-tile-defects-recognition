import math
import cv2
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QLabel
from ui_mainwindow import Ui_MainWindow
from ultralytics import YOLO
from window.sliding_window import sliding_window


def get_latest_save_dir():
    save_dir_list = [item for item in Path("runs/detect").iterdir() if item.is_dir()]
    if save_dir_list:
        sorted_save_dir = sorted(save_dir_list, key=lambda folder: folder.stat().st_ctime)
        return str(sorted_save_dir[-1])
    else:
        return


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, app):
        super().__init__()
        self.setupUi(self)
        self.app = app
        self.image_paths = []
        self.yolo_model = None
        self.predict_config = {
            'model_path': "models/based-models/best.pt",
            'image_path': "img_path",
            'conf': 0.2,
            'iou': 0.65
        }
        self.predict_mode = self.comboBox_choose_model.currentIndex()
        self.flag_already_predicted = False

        # menubar area
        self.actionQuit.triggered.connect(self.quit)
        self.actionAbout_Project.triggered.connect(self.about)
        self.actionAbout_Qt.triggered.connect(self.about_qt)

        # display image area
        self.label_original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_predicted_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # setup area
        self.pushButton_choose_source.clicked.connect(self.select_multi_image)
        self.comboBox_choose_model.currentTextChanged.connect(self.handle_model)

        self.slider_conf.setMinimum(1)
        self.slider_conf.setMaximum(100)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(lambda value: self.label_conf_value.setText(str(value/100)))
        self.slider_conf.sliderReleased.connect(self.change_slider_conf)
        self.slider_iou.setMinimum(1)
        self.slider_iou.setMaximum(95)
        self.slider_iou.setValue(65)
        self.slider_iou.valueChanged.connect(lambda value: self.label_iou_value.setText(str(value/100)))
        self.slider_iou.sliderReleased.connect(self.change_slider_iou)

        # Run prediction
        self.pushButton_run_prediction.clicked.connect(self.run_prediction)
        self.pushButton_next_image.setEnabled(False)
        self.pushButton_next_image.clicked.connect(self.select_next_image)
        self.pushButton_previous_image.setEnabled(False)
        self.pushButton_previous_image.clicked.connect(self.select_prev_image)

        # display label.txt area
        self.textEdit_show_label.setReadOnly(True)

    def quit(self):
        self.app.quit()

    def about(self):
        QMessageBox.information(self, "Project: ", "Ceramic Tile Defects Detection.")

    @staticmethod
    def about_qt():
        QApplication.aboutQt()

    def select_multi_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            self.image_paths = file_dialog.selectedFiles()
            self.predict_config['image_path'] = self.image_paths[0]
            self.show_image(self.label_original_image, self.image_paths[0])
            print(f"Source image has been selected. Path: {self.image_paths[0]}")
        else:
            print("No image selected, please select source image again!")

    def select_next_image(self):
        current_img_path_index = self.image_paths.index(self.predict_config['image_path'])
        next_image = self.image_paths[current_img_path_index + 1]
        self.predict_config['image_path'] = next_image
        self.show_image(self.label_original_image, next_image)
        if self.flag_already_predicted:
            self.run_prediction()

    def select_prev_image(self):
        current_img_path_index = self.image_paths.index(self.predict_config['image_path'])
        self.predict_config['image_path'] = self.image_paths[current_img_path_index - 1]
        self.show_image(self.label_original_image, self.predict_config['image_path'])
        if self.flag_already_predicted:
            self.run_prediction()

    def update_button_states(self):
        current_img_path_index = self.image_paths.index(self.predict_config['image_path'])
        self.pushButton_next_image.setEnabled(current_img_path_index < len(self.image_paths) - 1)
        self.pushButton_previous_image.setEnabled(current_img_path_index > 0)

    def show_image(self, label_image: QLabel, image_path: str):
        pixmap = QPixmap(image_path).scaled(label_image.size(),
                                            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                                            transformMode=Qt.TransformationMode.SmoothTransformation)
        label_image.setPixmap(pixmap)
        # label_image.setScaledContents(True)
        self.update_button_states()

    def handle_model(self):
        selected = self.comboBox_choose_model.currentIndex()
        if selected == 0:
            self.predict_config['model_path'] = r"models/based-models/best.pt"
        if selected == 1:
            self.predict_config['model_path'] = r"models/tiling-models/best.pt"
        self.yolo_model = YOLO(model=self.predict_config['model_path'])
        self.predict_mode = selected
        self.flag_already_predicted = False
        print(f"Model with {self.comboBox_choose_model.currentText()} has been selected.")

    def change_slider_conf(self):
        print("Confidence threshold = ", self.slider_conf.value())
        self.predict_config['conf'] = self.slider_conf.value() / 100

    def change_slider_iou(self):
        print("IoU threshold = ", self.slider_iou.value())
        self.predict_config['iou'] = self.slider_iou.value() / 100

    def show_label_content(self, save_dir: str):
        file_stem = Path(self.predict_config['image_path']).stem
        file_label = Path(f"{save_dir}/labels/{file_stem}.txt")
        if file_label.exists():
            with open(file_label, "r") as file:
                file_contents = file.read()
            self.textEdit_show_label.setPlainText(file_contents)
        else:
            pass

    def delete_current_label_file(self):
        if get_latest_save_dir() is None:
            return
        else:
            save_dir = get_latest_save_dir()
            label_file = Path(f"{save_dir}/labels/{str(Path(self.predict_config['image_path']).stem)}.txt")
            if label_file.exists():
                label_file.unlink()

    def run_prediction(self):
        if self.yolo_model is None:
            print("No model is selected")
            return
        else:
            self.yolo_model.info(verbose=True)

        self.delete_current_label_file()

        if self.predict_mode == 0:
            self.predict_default()
        else:
            self.predict_sliding()

        save_dir = get_latest_save_dir()
        predicted_image = f"{save_dir}/{Path(self.predict_config['image_path']).name}"
        self.show_image(self.label_predicted_image, predicted_image)
        self.show_label_content(save_dir)
        self.flag_already_predicted = True

    def predict_default(self):
        self.yolo_model(source=self.predict_config['image_path'], device=0, show=False,
                        conf=self.predict_config['conf'], iou=self.predict_config['iou'],
                        save=True, save_crop=False, save_txt=True, save_conf=True)

    def predict_sliding(self):
        img_path = Path(self.predict_config['image_path'])
        image = cv2.imread(str(img_path), 1)
        save_image = cv2.imread(str(img_path), 1)

        window_size = (600, 600)
        stride = 360
        save_dir = ""

        for x, y, window in sliding_window(image, window_size, stride):
            results = self.yolo_model(source=window, device=0, show=False,
                                      conf=self.predict_config['conf'], iou=self.predict_config['iou'],
                                      save=True, save_crop=False, save_txt=True, save_conf=True)
            save_dir = get_latest_save_dir()
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
                    if cls == "chipping":
                        cls = "edge-chipping"

                    # Draw bounding box, class name and confidence value on original image
                    cv2.rectangle(img=save_image,
                                  pt1=(x_abs, y_abs), pt2=(x_abs + w_obj, y_abs + h_obj),
                                  color=(31, 113, 255), thickness=5)

                    cv2.putText(img=save_image, text=f"{cls} - {conf}",
                                org=(max(0, x_abs), max(20, y_abs - 5)),
                                fontFace=cv2.FONT_ITALIC, fontScale=3,
                                color=(3, 6, 191), thickness=6)

        cv2.imwrite(f"{save_dir}/{img_path.name}", save_image)
        Path(f"{save_dir}/labels/image0.txt").rename(f"{save_dir}/labels/{img_path.stem}.txt")

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Source Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            # self.label_original_image.setProperty("image_path", file_path)
            # image_path = self.label_original_image.property("image_path")
            self.predict_config['image_path'] = file_path
            self.show_image(self.label_original_image, file_path)
            print(f"Source image has been selected. Path: {file_path}")
        else:
            print("No image selected, please select source image again!")
