# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt6 UI code generator 6.5.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setHorizontalSpacing(15)
        self.gridLayout.setVerticalSpacing(13)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_label_area = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame_label_area.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.frame_label_area.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_label_area.setObjectName("frame_label_area")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_label_area)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.textEdit_show_label = QtWidgets.QTextEdit(parent=self.frame_label_area)
        self.textEdit_show_label.setObjectName("textEdit_show_label")
        self.horizontalLayout_6.addWidget(self.textEdit_show_label)
        self.gridLayout.addWidget(self.frame_label_area, 1, 1, 1, 1)
        self.frame_image_area = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame_image_area.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.frame_image_area.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.frame_image_area.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_image_area.setLineWidth(1)
        self.frame_image_area.setMidLineWidth(0)
        self.frame_image_area.setObjectName("frame_image_area")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_image_area)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setHorizontalSpacing(35)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_original_image = QtWidgets.QLabel(parent=self.frame_image_area)
        self.label_original_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_original_image.setObjectName("label_original_image")
        self.gridLayout_3.addWidget(self.label_original_image, 0, 0, 1, 1)
        self.label_predicted_image = QtWidgets.QLabel(parent=self.frame_image_area)
        self.label_predicted_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_predicted_image.setObjectName("label_predicted_image")
        self.gridLayout_3.addWidget(self.label_predicted_image, 0, 1, 1, 1)
        self.horizontalLayout_7.addLayout(self.gridLayout_3)
        self.gridLayout.addWidget(self.frame_image_area, 0, 0, 1, 2)
        self.frame_setup_area = QtWidgets.QFrame(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_setup_area.sizePolicy().hasHeightForWidth())
        self.frame_setup_area.setSizePolicy(sizePolicy)
        self.frame_setup_area.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.frame_setup_area.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_setup_area.setObjectName("frame_setup_area")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_setup_area)
        self.horizontalLayout_2.setSpacing(15)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_source = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_source.setObjectName("label_source")
        self.gridLayout_2.addWidget(self.label_source, 0, 0, 1, 1)
        self.label_model = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_model.setObjectName("label_model")
        self.gridLayout_2.addWidget(self.label_model, 1, 0, 1, 1)
        self.label_winsize = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_winsize.setObjectName("label_winsize")
        self.gridLayout_2.addWidget(self.label_winsize, 2, 0, 1, 2)
        self.lineEdit_set_winside = QtWidgets.QLineEdit(parent=self.frame_setup_area)
        self.lineEdit_set_winside.setObjectName("lineEdit_set_winside")
        self.gridLayout_2.addWidget(self.lineEdit_set_winside, 2, 2, 1, 1)
        self.comboBox_choose_model = QtWidgets.QComboBox(parent=self.frame_setup_area)
        self.comboBox_choose_model.setObjectName("comboBox_choose_model")
        self.comboBox_choose_model.addItem("")
        self.comboBox_choose_model.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_choose_model, 1, 2, 1, 1)
        self.pushButton_choose_source = QtWidgets.QPushButton(parent=self.frame_setup_area)
        self.pushButton_choose_source.setObjectName("pushButton_choose_source")
        self.gridLayout_2.addWidget(self.pushButton_choose_source, 0, 2, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_conf = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_conf.setObjectName("label_conf")
        self.verticalLayout.addWidget(self.label_conf)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.slider_conf = QtWidgets.QSlider(parent=self.frame_setup_area)
        self.slider_conf.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_conf.setObjectName("slider_conf")
        self.horizontalLayout_3.addWidget(self.slider_conf)
        self.label_conf_value = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_conf_value.setObjectName("label_conf_value")
        self.horizontalLayout_3.addWidget(self.label_conf_value)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_iou = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_iou.setObjectName("label_iou")
        self.verticalLayout_2.addWidget(self.label_iou)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.slider_iou = QtWidgets.QSlider(parent=self.frame_setup_area)
        self.slider_iou.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider_iou.setObjectName("slider_iou")
        self.horizontalLayout_4.addWidget(self.slider_iou)
        self.label_iou_value = QtWidgets.QLabel(parent=self.frame_setup_area)
        self.label_iou_value.setObjectName("label_iou_value")
        self.horizontalLayout_4.addWidget(self.label_iou_value)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_run_prediction = QtWidgets.QPushButton(parent=self.frame_setup_area)
        self.pushButton_run_prediction.setObjectName("pushButton_run_prediction")
        self.verticalLayout_4.addWidget(self.pushButton_run_prediction)
        self.pushButton_next_image = QtWidgets.QPushButton(parent=self.frame_setup_area)
        self.pushButton_next_image.setObjectName("pushButton_next_image")
        self.verticalLayout_4.addWidget(self.pushButton_next_image)
        self.pushButton_previous_image = QtWidgets.QPushButton(parent=self.frame_setup_area)
        self.pushButton_previous_image.setObjectName("pushButton_previous_image")
        self.verticalLayout_4.addWidget(self.pushButton_previous_image)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 5)
        self.horizontalLayout_2.setStretch(2, 1)
        self.gridLayout.addWidget(self.frame_setup_area, 1, 0, 1, 1)
        self.gridLayout.setRowStretch(0, 10)
        self.gridLayout.setRowStretch(1, 2)
        self.horizontalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(parent=self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuMode = QtWidgets.QMenu(parent=self.menubar)
        self.menuMode.setObjectName("menuMode")
        self.menuHelp = QtWidgets.QMenu(parent=self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(parent=MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExample = QtGui.QAction(parent=MainWindow)
        self.actionExample.setObjectName("actionExample")
        self.actionQuit = QtGui.QAction(parent=MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionmode1 = QtGui.QAction(parent=MainWindow)
        self.actionmode1.setObjectName("actionmode1")
        self.actionmode2 = QtGui.QAction(parent=MainWindow)
        self.actionmode2.setObjectName("actionmode2")
        self.actionmode3 = QtGui.QAction(parent=MainWindow)
        self.actionmode3.setObjectName("actionmode3")
        self.actionmode4 = QtGui.QAction(parent=MainWindow)
        self.actionmode4.setObjectName("actionmode4")
        self.actionmode5 = QtGui.QAction(parent=MainWindow)
        self.actionmode5.setObjectName("actionmode5")
        self.actionHelp = QtGui.QAction(parent=MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionDocument = QtGui.QAction(parent=MainWindow)
        self.actionDocument.setObjectName("actionDocument")
        self.actionAbout_Project = QtGui.QAction(parent=MainWindow)
        self.actionAbout_Project.setObjectName("actionAbout_Project")
        self.actionAbout_Qt = QtGui.QAction(parent=MainWindow)
        self.actionAbout_Qt.setObjectName("actionAbout_Qt")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionExample)
        self.menuFile.addAction(self.actionQuit)
        self.menuMode.addAction(self.actionmode1)
        self.menuMode.addAction(self.actionmode2)
        self.menuMode.addSeparator()
        self.menuMode.addAction(self.actionmode3)
        self.menuMode.addAction(self.actionmode4)
        self.menuMode.addAction(self.actionmode5)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addAction(self.actionDocument)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout_Project)
        self.menuHelp.addAction(self.actionAbout_Qt)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_original_image.setText(_translate("MainWindow", "original_image"))
        self.label_predicted_image.setText(_translate("MainWindow", "predicted_image"))
        self.label_source.setText(_translate("MainWindow", "Source "))
        self.label_model.setText(_translate("MainWindow", "Model "))
        self.label_winsize.setText(_translate("MainWindow", "Window-size"))
        self.comboBox_choose_model.setItemText(0, _translate("MainWindow", "Baseline model"))
        self.comboBox_choose_model.setItemText(1, _translate("MainWindow", "Sliding window"))
        self.pushButton_choose_source.setText(_translate("MainWindow", "choose source"))
        self.label_conf.setText(_translate("MainWindow", "Confidence Threshold"))
        self.label_conf_value.setText(_translate("MainWindow", "0.25"))
        self.label_iou.setText(_translate("MainWindow", "IoU Threshold"))
        self.label_iou_value.setText(_translate("MainWindow", "0.65"))
        self.pushButton_run_prediction.setText(_translate("MainWindow", "Predict"))
        self.pushButton_next_image.setText(_translate("MainWindow", "Next image"))
        self.pushButton_previous_image.setText(_translate("MainWindow", "Prev image"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuMode.setTitle(_translate("MainWindow", "Mode"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionExample.setText(_translate("MainWindow", "Example"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionmode1.setText(_translate("MainWindow", "mode1"))
        self.actionmode2.setText(_translate("MainWindow", "mode2"))
        self.actionmode3.setText(_translate("MainWindow", "mode3"))
        self.actionmode4.setText(_translate("MainWindow", "mode4"))
        self.actionmode5.setText(_translate("MainWindow", "mode5"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionDocument.setText(_translate("MainWindow", "Document"))
        self.actionAbout_Project.setText(_translate("MainWindow", "About Project"))
        self.actionAbout_Qt.setText(_translate("MainWindow", "About Qt"))