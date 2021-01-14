# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDate
import mysql.connector as mc
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox
import cv2
import os
from datetime import datetime

from config import *
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QPixmap

import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import re 

sys.path.append('E:\LVTN\Sample\Faster_RCNN_for_Open_Images_Dataset_Keras-master/')
from prediction import *

video_path = 'F:\LVTN\Test_Video/test_CAM3.mp4'


class Ui_MainWindow(object):
    """
    Main class of UI
    """
    path_file = None
    sdt = None
    email = None
    predict_video = Predict_video()
    predict_video.set_up_scenario()

    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(804, 477)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 801, 451))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 5, 3, 1, 1)
        self.btn_play = QtWidgets.QPushButton(self.tab)
        self.btn_play.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_play.setFont(font)
        self.btn_play.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_play.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                    "font: 600 13pt \"MS Shell Dlg 2\";\n"
                                    "border-color: rgb(85, 0, 255);\n"
                                    "color: rgb(255, 255, 255);")
        self.btn_play.setObjectName("btn_play")
        self.gridLayout.addWidget(self.btn_play, 5, 2, 1, 1)
        self.btn_reset = QtWidgets.QPushButton(self.tab)
        self.btn_reset.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_reset.setFont(font)
        self.btn_reset.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                    "font: 600 13pt \"MS Shell Dlg 2\";\n"
                                    "border-color: rgb(85, 0, 255);\n"
                                    "color: rgb(255, 255, 255);")
        self.btn_reset.setObjectName("btn_reset")
        self.gridLayout.addWidget(self.btn_reset, 5, 4, 1, 1)
        self.label_auth = QtWidgets.QLabel(self.tab)
        self.label_auth.setObjectName("label_auth")
        self.gridLayout.addWidget(self.label_auth, 2, 0, 1, 1)
        self.label_show_path_file = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_show_path_file.sizePolicy().hasHeightForWidth())
        self.label_show_path_file.setSizePolicy(sizePolicy)
        self.label_show_path_file.setText("")
        self.label_show_path_file.setObjectName("label_show_path_file")
        self.gridLayout.addWidget(self.label_show_path_file, 7, 2, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 5, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 8, 0, 1, 1)
        self.label_path_file = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_path_file.sizePolicy().hasHeightForWidth())
        self.label_path_file.setSizePolicy(sizePolicy)
        self.label_path_file.setMinimumSize(QtCore.QSize(0, 20))
        self.label_path_file.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_path_file.setObjectName("label_path_file")
        self.gridLayout.addWidget(self.label_path_file, 7, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 9, 0, 1, 1)
        self.txt_email = QtWidgets.QTextEdit(self.tab)
        self.txt_email.setMinimumSize(QtCore.QSize(0, 30))
        self.txt_email.setMaximumSize(QtCore.QSize(306, 30))
        self.txt_email.setObjectName("txt_email")
        self.gridLayout.addWidget(self.txt_email, 9, 2, 1, 1)
        self.btn_from_file = QtWidgets.QPushButton(self.tab)
        self.btn_from_file.setMinimumSize(QtCore.QSize(0, 50))
        self.btn_from_file.setSizeIncrement(QtCore.QSize(0, 900))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_from_file.setFont(font)
        self.btn_from_file.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                        "font: 600 13pt \"MS Shell Dlg 2\";\n"
                                        "border-color: rgb(85, 0, 255);\n"
                                        "color: rgb(255, 255, 255);")
        self.btn_from_file.setObjectName("btn_from_file")
        self.gridLayout.addWidget(self.btn_from_file, 5, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        self.label_title.setMaximumSize(QtCore.QSize(16777215, 42))
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 5)
        self.label_mentor = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_mentor.sizePolicy().hasHeightForWidth())
        self.label_mentor.setSizePolicy(sizePolicy)
        self.label_mentor.setMaximumSize(QtCore.QSize(16777215, 38))
        self.label_mentor.setObjectName("label_mentor")
        self.gridLayout.addWidget(self.label_mentor, 2, 4, 1, 1)
        self.txt_sdt = QtWidgets.QTextEdit(self.tab)
        self.txt_sdt.setMinimumSize(QtCore.QSize(0, 30))
        self.txt_sdt.setMaximumSize(QtCore.QSize(306, 30))
        self.txt_sdt.setObjectName("txt_sdt")
        self.gridLayout.addWidget(self.txt_sdt, 8, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout.addItem(spacerItem2, 3, 0, 1, 1)
        self.btnConfig = QtWidgets.QPushButton(self.tab)
        self.btnConfig.setMinimumSize(QtCore.QSize(192, 50))
        self.btnConfig.setStyleSheet("background-color: rgb(85, 170, 255);\n"
                                    "font: 600 13pt \"MS Shell Dlg 2\";\n"
                                    "border-color: rgb(85, 0, 255);\n"
                                    "color: rgb(255, 255, 255);")
        self.btnConfig.setObjectName("btnConfig")
        self.gridLayout.addWidget(self.btnConfig, 8, 4, 2, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout.addItem(spacerItem3, 6, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 3, 1, 1)
        self.txtSearch = QtWidgets.QLineEdit(self.tab_2)
        self.txtSearch.setObjectName("txtSearch")
        self.gridLayout_2.addWidget(self.txtSearch, 2, 0, 1, 1)
        self.btn_filter = QtWidgets.QPushButton(self.tab_2)
        self.btn_filter.setStyleSheet("background-color: rgb(0, 0, 255);\n"
                                    "border-color: rgb(255, 0, 0);\n"
                                    "color: rgb(255, 255, 255);\n"
                                    "font: 75 10pt \"MS Shell Dlg 2\";")
        self.btn_filter.setObjectName("btn_filter")
        self.gridLayout_2.addWidget(self.btn_filter, 2, 7, 1, 1)
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 8)
        spacerItem4 = QtWidgets.QSpacerItem(150, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 2, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 2, 5, 1, 1)
        self.fromDate = QtWidgets.QDateEdit(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fromDate.sizePolicy().hasHeightForWidth())
        self.fromDate.setSizePolicy(sizePolicy)
        self.fromDate.setMinimumSize(QtCore.QSize(100, 0))
        self.fromDate.setAlignment(QtCore.Qt.AlignCenter)
        self.fromDate.setCalendarPopup(True)
        self.fromDate.setCurrentSectionIndex(0)
        self.fromDate.setDate(QtCore.QDate(2020, 12, 12))
        self.fromDate.setObjectName("fromDate")
        self.gridLayout_2.addWidget(self.fromDate, 2, 4, 1, 1)
        self.btnSearch = QtWidgets.QPushButton(self.tab_2)
        self.btnSearch.setStyleSheet("background-color: rgb(0, 0, 255);\n"
                                    "border-color: rgb(255, 0, 0);\n"
                                    "color: rgb(255, 255, 255);\n"
                                    "font: 75 10pt \"MS Shell Dlg 2\";")
        self.btnSearch.setObjectName("btnSearch")
        self.gridLayout_2.addWidget(self.btnSearch, 2, 1, 1, 1)
        self.tableWidget = QtWidgets.QTableWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(6, item)
        self.gridLayout_2.addWidget(self.tableWidget, 3, 0, 1, 8)
        self.toDate = QtWidgets.QDateEdit(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toDate.sizePolicy().hasHeightForWidth())
        self.toDate.setSizePolicy(sizePolicy)
        self.toDate.setMinimumSize(QtCore.QSize(100, 0))
        self.toDate.setAlignment(QtCore.Qt.AlignCenter)
        self.toDate.setCalendarPopup(True)
        self.toDate.setDate(QtCore.QDate(2020, 12, 12))
        self.toDate.setObjectName("toDate")
        self.gridLayout_2.addWidget(self.toDate, 2, 6, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.gridLayout_2.addItem(spacerItem5, 1, 0, 1, 8)
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Set up at begining
        self.label_show_path_file.setText(video_path)
        self.txt_email.setText('nhattaia6@gmail.com')
        self.txt_sdt.setText('0338684430')

        # my edit
        self.event_processing()

        # Init config UI
        self.config_Dialog = QtWidgets.QDialog()
        self.config_Dialog.setWindowTitle('Config')
        self.config_ui = Ui_config_Dialog()
        self.config_ui.setupUi(self.config_Dialog)

        self.process_image_every_x_frame     = self.config_ui.processImageEveryXFrameSpinBox.value()
        self.create_new_thread_every_x_frame = self.config_ui.createNewThreadEveryXFrameSpinBox.value()
        self.check_result_every_x_frame      = self.config_ui.checkResultEveryXFrameSpinBox.value()
        self.send_result_every_x_frame       = self.config_ui.sendResultEveryXFrameSpinBox.value()
        self.create_thread_for_x_frame_first = self.config_ui.createThreadForXFrameFirstSpinBox.value()
        self.rotate_image                    = self.config_ui.rotateImageSpinBox.value()
        self.send_sms                        = self.config_ui.sendSmsCheckBox.isChecked()
        self.send_email                      = self.config_ui.sendEmailCheckBox.isChecked()
        self.ratio_car_per_bike              = self.config_ui.ratioCarBikeSpinBox.value()
        self.ratio_priority_per_bike         = self.config_ui.ratioPriorityBikeSpinBox.value()
        self.min_status                      = self.config_ui.minSpinBox.value()
        self.max_status                      = self.config_ui.maxSpinBox.value()


        # init value for table
        self.dbname = 'traffic_density_estimation_log'
        self.tablename = 'logs'

        sql = "SELECT date, time, bike, car, priority, status, image FROM {} ".format(self.tablename)
        self.get_data(sql)

        # set time to today
        self.fromDate.setDate(QDate.currentDate())
        self.toDate.setDate(QDate.currentDate())

        # Regex for email
        self.email_regex = r'^[\w]+@([\w-]+\.)+[\w-]{2,4}$'


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Traffic Density Estimation System"))
        self.btn_play.setText(_translate("MainWindow", "PLAY"))
        self.btn_play.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.btn_reset.setText(_translate("MainWindow", "RESET"))
        self.btn_reset.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.label_auth.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600;\">Auth: Bui Nhat Tai - B1606838</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt; font-weight:600; color:#5500ff;\">Phone number</span></p></body></html>"))
        self.label_path_file.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt; font-weight:600; color:#5500ff;\">File path: </span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt; font-weight:600; color:#5500ff;\">Email</span></p></body></html>"))
        self.btn_from_file.setText(_translate("MainWindow", "FROM FILE"))
        self.btn_from_file.setShortcut(_translate("MainWindow", "Ctrl+F"))
        self.label_title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:600; color:#00de00;\">TRAFFIC DENSITY ESTIMATION</span></p></body></html>"))
        self.label_mentor.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">Mentor: TS. Tran Nguyen Minh Thu</span></p><p align=\"center\"><span style=\" font-weight:600;\">NCS. Vu Le Quynh Phuong</span></p></body></html>"))
        self.btnConfig.setText(_translate("MainWindow", "Config"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Traffic Density Estimation System"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color:#0055ff;\">FROM</span></p></body></html>"))
        self.txtSearch.setPlaceholderText(_translate("MainWindow", "Search..."))
        self.btn_filter.setText(_translate("MainWindow", "Filter"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; color:#5555ff;\">Traffic Density Estimation Log</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color:#5555ff;\">TO</span></p></body></html>"))
        self.fromDate.setDisplayFormat(_translate("MainWindow", "dd/MM/yyyy"))
        self.btnSearch.setText(_translate("MainWindow", "Search"))
        self.tableWidget.setSortingEnabled(True)
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Date"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Bike"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Car"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Priority"))
        item = self.tableWidget.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Status"))
        item = self.tableWidget.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Traffic Density Estimation Logs"))

        # Link d-click event to open image function
        self.tableWidget.itemDoubleClicked.connect(self.open_image)
        self.btnSearch.clicked.connect(self.search_event)
        self.btn_filter.clicked.connect(self.filter_event)
        
        self.tabWidget.tabBarClicked.connect(self.switch_tab_event)

    ############# Function of system #############
    #Check valid email
    def check_email(self,email): 
        """
        Check email address with regex
        """
        if(re.search(self.email_regex,email)):
            print("Valid Email") 
            return True
        else:
            print("Invalid Email") 
            return False

    # Config event
    def config_event(self):
        # Set current value
        self.config_ui.processImageEveryXFrameSpinBox.setValue(self.process_image_every_x_frame)
        self.config_ui.createNewThreadEveryXFrameSpinBox.setValue(self.create_new_thread_every_x_frame)
        self.config_ui.checkResultEveryXFrameSpinBox.setValue(self.check_result_every_x_frame)
        self.config_ui.sendResultEveryXFrameSpinBox.setValue(self.send_result_every_x_frame)
        self.config_ui.createThreadForXFrameFirstSpinBox.setValue(self.create_thread_for_x_frame_first)
        self.config_ui.rotateImageSpinBox.setValue(self.rotate_image)
        self.config_ui.sendSmsCheckBox.setChecked(self.send_sms)
        self.config_ui.sendEmailCheckBox.setChecked(self.send_email)
        self.config_ui.ratioCarBikeSpinBox.setValue(self.ratio_car_per_bike)
        self.config_ui.ratioPriorityBikeSpinBox.setValue(self.ratio_priority_per_bike)
        self.config_ui.minSpinBox.setValue(self.min_status)
        self.config_ui.maxSpinBox.setValue(self.max_status)

        self.config_Dialog.show()
        if self.config_Dialog.exec_():
            self.process_image_every_x_frame     = self.config_ui.processImageEveryXFrameSpinBox.value()
            self.create_new_thread_every_x_frame = self.config_ui.createNewThreadEveryXFrameSpinBox.value()
            self.check_result_every_x_frame      = self.config_ui.checkResultEveryXFrameSpinBox.value()
            self.send_result_every_x_frame       = self.config_ui.sendResultEveryXFrameSpinBox.value()
            self.create_thread_for_x_frame_first = self.config_ui.createThreadForXFrameFirstSpinBox.value()
            self.rotate_image                    = self.config_ui.rotateImageSpinBox.value()
            self.send_sms                        = self.config_ui.sendSmsCheckBox.isChecked()
            self.send_email                      = self.config_ui.sendEmailCheckBox.isChecked()
            self.ratio_car_per_bike              = self.config_ui.ratioCarBikeSpinBox.value()
            self.ratio_priority_per_bike         = self.config_ui.ratioPriorityBikeSpinBox.value()
            self.min_status                      = self.config_ui.minSpinBox.value()
            self.max_status                      = self.config_ui.maxSpinBox.value()
            
            if self.min_status >= self.max_status:
                QMessageBox.warning(self.tableWidget, 'Error', 'Min value must be less than Max value!')
            else:
                print("Success!")
                print("self.process_image_every_x_frame: "   ,self.process_image_every_x_frame)
                print("self.create_new_thread_every_x_frame: "   ,self.create_new_thread_every_x_frame)
                print("self.check_result_every_x_frame: "      ,self.check_result_every_x_frame)
                print("self.send_result_every_x_frame: "     ,self.send_result_every_x_frame)
                print("self.create_thread_for_x_frame_first: ",self.create_thread_for_x_frame_first)
                print("self.rotate_image: ",self.rotate_image)
                print("self.send_sms: ",self.send_sms)
                print("self.send_email: ",self.send_email)
                print("self.ratio_car_per_bike: ",self.ratio_car_per_bike)
                print("self.ratio_priority_per_bike: ",self.ratio_priority_per_bike)
                print("self.min_status: ",self.min_status)
                print("self.max_status: ",self.max_status)
                self.predict_video.set_up_scenario(self.process_image_every_x_frame, 
                                                    self.create_new_thread_every_x_frame,
                                                    self.check_result_every_x_frame, 
                                                    self.send_result_every_x_frame, 
                                                    self.create_thread_for_x_frame_first,
                                                    self.rotate_image,
                                                    self.send_sms,
                                                    self.send_email,
                                                    self.min_status,
                                                    self.max_status,
                                                    self.ratio_car_per_bike,
                                                    self.ratio_priority_per_bike)
        else:
            print("Cancel!")

    def event_processing(self):
        """
        Connect buttons and events
        """
        self.btn_from_file.clicked.connect(self.from_file)
        self.btn_play.clicked.connect(self.make_processing)
        self.btn_reset.clicked.connect(self.reset_event)
        self.btnConfig.clicked.connect(self.config_event)

    def from_file(self):
        """
        Get video file path
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"Open video", "F:\LVTN\Test_Video","Video Files (*.mp4 *.avi *.flv *.wmv *.mov)", options=options)
        if fileName:
            print(fileName)
            self.path_file = fileName
            self.label_path_file.setVisible(True)
            self.label_show_path_file.setText(fileName)
    
    # def make_processing(self, file_path):
    #     def processing():
    #         print("here is path file!", file_path)
    #     return processing

    def make_processing(self):
        """
        Start process video
        """
        _translate = QtCore.QCoreApplication.translate
        self.sdt = self.txt_sdt.toPlainText()
        self.email = self.txt_email.toPlainText()
        self.path_file = self.label_show_path_file.text()
        print("SDT", self.sdt)
        print("EMAIL", self.email)
        

        if self.send_email:
            if self.email:
                mail_list = self.email.split(';')
                mail_list = list(map(str.strip,mail_list))
                valid_mail_list = [mail for mail in mail_list if self.check_email(mail)]
                print("Email list", valid_mail_list)
                if valid_mail_list:
                    if self.send_sms:
                        if self.sdt:
                            if not self.sdt.isdigit():
                                QMessageBox.warning(self.tableWidget, 'Error', 'Incorrect phone number!')
                            else:
                                if os.path.isfile(self.path_file):
                                    print("here is path file!", self.path_file)

                                    self.predict_video.init_video(self.path_file, self.sdt, valid_mail_list, bbox_threshold=0.5)
                                    if self.predict_video.get_counter_area() == False:
                                        QMessageBox.warning(self.tableWidget, 'Warning', 'Cannot get counter area!')
                                    else:
                                        self.predict_video.run_predict_video()
                                else:
                                    print("File not found!")               
                                    QMessageBox.warning(self.tableWidget, 'Error', 'File not found! Please, try again!')
                        else:
                            QMessageBox.warning(self.tableWidget, 'Error', 'Phone number must not empty! Please, try again!')
                    else:
                        if os.path.isfile(self.path_file):
                            print("here is path file!", self.path_file)

                            self.predict_video.init_video(self.path_file, email=valid_mail_list, bbox_threshold=0.5)
                            if self.predict_video.get_counter_area() == False:
                                QMessageBox.warning(self.tableWidget, 'Warning', 'Cannot get counter area!')
                            else:
                                self.predict_video.run_predict_video()
                        else:
                            print("File not found!")               
                            QMessageBox.warning(self.tableWidget, 'Error', 'File not found! Please, try again!')
                else:
                    QMessageBox.warning(self.tableWidget, 'Error', 'Invalid email address! Please, try again!')
            else:
                QMessageBox.warning(self.tableWidget, 'Error', 'Email address must not empty! Please, try again!')
        elif self.send_sms:
            if self.sdt:
                if not self.sdt.isdigit():
                    QMessageBox.warning(self.tableWidget, 'Error', 'Incorrect phone number!')
                else:
                    if os.path.isfile(self.path_file):
                        print("here is path file!", self.path_file)

                        self.predict_video.init_video(self.path_file, self.sdt, bbox_threshold=0.5)
                        if self.predict_video.get_counter_area() == False:
                            QMessageBox.warning(self.tableWidget, 'Warning', 'Cannot get counter area!')
                        else:
                            self.predict_video.run_predict_video()
                    else:
                        print("File not found!")               
                        QMessageBox.warning(self.tableWidget, 'Error', 'File not found! Please, try again!')
            else:
                QMessageBox.warning(self.tableWidget, 'Error', 'Phone number must not empty! Please, try again!')
        else:
            if os.path.isfile(self.path_file):
                print("here is path file!", self.path_file)

                self.predict_video.init_video(self.path_file, bbox_threshold=0.5)
                if self.predict_video.get_counter_area() == False:
                    QMessageBox.warning(self.tableWidget, 'Warning', 'Cannot get counter area!')
                else:
                    self.predict_video.run_predict_video()
            else:
                print("File not found!")               
                QMessageBox.warning(self.tableWidget, 'Error', 'File not found! Please, try again!')

    def reset_event(self):
        """
        Reset event processing
        """
        self.path_file = None
        self.label_path_file.setVisible(False)
        self.label_show_path_file.setText('')
        self.txt_email.setText('')
        self.txt_sdt.setText('')

    ############## Function of Logs #############
    def switch_tab_event(self):
        sql = "SELECT date, time, bike, car, priority, status, image FROM {} ".format(self.tablename)
        self.get_data(sql)

    def rescale_frame(self, frame, percent=75):
        """
        Scale frame with X percent (default 75)
        return frame after scaled
        """
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
        return frame

    def open_image(self,item):
        """
        Open image from image path
        """
        if item.column() == 6:
            print("path file:", item.text())
            if os.path.isfile(item.text()):
                img = cv2.imread(item.text())
                img = cv2.resize(img, (1280,720))
                img = self.rescale_frame(img)
                cv2.imshow("Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("File not found!")
                QMessageBox.about(self.tableWidget, 'Warning', 'File not found!')

    def get_data(self, sql):
        """
        Get data with sql command
        """
        try:      
            self.tableWidget.setSortingEnabled(False)

            mydb = mc.connect(
                host="localhost",
                user="admin",
                password="Coincard2@",
                database=self.dbname
            )
            mycursor = mydb.cursor()
            mycursor.execute(sql)
 
            result = mycursor.fetchall()
            self.tableWidget.setRowCount(0)
            for row_number, row_data in enumerate(result):
                # print(row_number)
                self.tableWidget.insertRow(row_number)
                for column_number, data in enumerate(row_data):
                    self.tableWidget.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
                    # print(column_number)                    
                    self.tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(data)))
                    if column_number==6:                        
                        self.tableWidget.setItem(row_number, column_number, QTableWidgetItem(str(data)))
                        self.tableWidget.item(row_number, column_number).setForeground(QtGui.QColor(0,0,255))
        except mc.Error as e:
            print("Error:", e)
        self.tableWidget.setSortingEnabled(True)

    def search_event(self):
        """
        Process search event
        """
        search_text = self.txtSearch.text()
        print("Search: ",search_text)
        
        sql = "SELECT date, time, bike, car, priority, status, image FROM {} \
                WHERE status like '%{}%'\
                ".format(self.tablename, search_text)

        self.get_data(sql)

    def filter_event(self):
        """
        Process filter event
        """
        from_date = datetime.strptime(self.fromDate.text(), '%d/%m/%Y').strftime('%Y-%m-%d')
        to_date = datetime.strptime(self.toDate.text(), '%d/%m/%Y').strftime('%Y-%m-%d')
        print("Filter from date {} to date {}:".format(from_date, to_date))

        sql = "SELECT date, time, bike, car, priority, status, image FROM {} \
                WHERE date >= '{}' and date <= '{}'\
                ".format(self.tablename, from_date, to_date)

        self.get_data(sql)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())