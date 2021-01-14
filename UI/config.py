# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'configv2.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_config_Dialog(object):
    def setupUi(self, config_Dialog):
        config_Dialog.setObjectName("config_Dialog")
        config_Dialog.resize(724, 480)
        self.buttonBox = QtWidgets.QDialogButtonBox(config_Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(320, 420, 341, 32))
        self.buttonBox.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(85, 85, 255);\n"
"border-color: rgb(255, 0, 0);\n"
"font: 75 14pt \"MS Shell Dlg 2\";")
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayoutWidget = QtWidgets.QWidget(config_Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 710, 381))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.processImageEveryXFrameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.processImageEveryXFrameLabel.setObjectName("processImageEveryXFrameLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.processImageEveryXFrameLabel)
        self.processImageEveryXFrameSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processImageEveryXFrameSpinBox.sizePolicy().hasHeightForWidth())
        self.processImageEveryXFrameSpinBox.setSizePolicy(sizePolicy)
        self.processImageEveryXFrameSpinBox.setMinimumSize(QtCore.QSize(100, 0))
        self.processImageEveryXFrameSpinBox.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.processImageEveryXFrameSpinBox.setFont(font)
        self.processImageEveryXFrameSpinBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.processImageEveryXFrameSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.processImageEveryXFrameSpinBox.setMinimum(1)
        self.processImageEveryXFrameSpinBox.setMaximum(999999999)
        self.processImageEveryXFrameSpinBox.setObjectName("processImageEveryXFrameSpinBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.processImageEveryXFrameSpinBox)
        self.createNewThreadEveryXFrameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.createNewThreadEveryXFrameLabel.setObjectName("createNewThreadEveryXFrameLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.createNewThreadEveryXFrameLabel)
        self.createNewThreadEveryXFrameSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.createNewThreadEveryXFrameSpinBox.sizePolicy().hasHeightForWidth())
        self.createNewThreadEveryXFrameSpinBox.setSizePolicy(sizePolicy)
        self.createNewThreadEveryXFrameSpinBox.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.createNewThreadEveryXFrameSpinBox.setFont(font)
        self.createNewThreadEveryXFrameSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.createNewThreadEveryXFrameSpinBox.setMinimum(1)
        self.createNewThreadEveryXFrameSpinBox.setMaximum(999999999)
        self.createNewThreadEveryXFrameSpinBox.setObjectName("createNewThreadEveryXFrameSpinBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.createNewThreadEveryXFrameSpinBox)
        self.checkResultEveryXFrameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.checkResultEveryXFrameLabel.setObjectName("checkResultEveryXFrameLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.checkResultEveryXFrameLabel)
        self.checkResultEveryXFrameSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkResultEveryXFrameSpinBox.sizePolicy().hasHeightForWidth())
        self.checkResultEveryXFrameSpinBox.setSizePolicy(sizePolicy)
        self.checkResultEveryXFrameSpinBox.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkResultEveryXFrameSpinBox.setFont(font)
        self.checkResultEveryXFrameSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.checkResultEveryXFrameSpinBox.setMinimum(1)
        self.checkResultEveryXFrameSpinBox.setMaximum(999999999)
        self.checkResultEveryXFrameSpinBox.setObjectName("checkResultEveryXFrameSpinBox")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.checkResultEveryXFrameSpinBox)
        self.sendResultEveryXFrameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.sendResultEveryXFrameLabel.setObjectName("sendResultEveryXFrameLabel")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.sendResultEveryXFrameLabel)
        self.sendResultEveryXFrameSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sendResultEveryXFrameSpinBox.sizePolicy().hasHeightForWidth())
        self.sendResultEveryXFrameSpinBox.setSizePolicy(sizePolicy)
        self.sendResultEveryXFrameSpinBox.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sendResultEveryXFrameSpinBox.setFont(font)
        self.sendResultEveryXFrameSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.sendResultEveryXFrameSpinBox.setMinimum(1)
        self.sendResultEveryXFrameSpinBox.setMaximum(999999999)
        self.sendResultEveryXFrameSpinBox.setObjectName("sendResultEveryXFrameSpinBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.sendResultEveryXFrameSpinBox)
        self.createThreadForXFrameFirstLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.createThreadForXFrameFirstLabel.setObjectName("createThreadForXFrameFirstLabel")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.createThreadForXFrameFirstLabel)
        self.createThreadForXFrameFirstSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.createThreadForXFrameFirstSpinBox.sizePolicy().hasHeightForWidth())
        self.createThreadForXFrameFirstSpinBox.setSizePolicy(sizePolicy)
        self.createThreadForXFrameFirstSpinBox.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.createThreadForXFrameFirstSpinBox.setFont(font)
        self.createThreadForXFrameFirstSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.createThreadForXFrameFirstSpinBox.setMinimum(1)
        self.createThreadForXFrameFirstSpinBox.setMaximum(999999999)
        self.createThreadForXFrameFirstSpinBox.setObjectName("createThreadForXFrameFirstSpinBox")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.createThreadForXFrameFirstSpinBox)
        self.rotateImageLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.rotateImageLabel.setObjectName("rotateImageLabel")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.rotateImageLabel)
        self.rotateImageSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rotateImageSpinBox.sizePolicy().hasHeightForWidth())
        self.rotateImageSpinBox.setSizePolicy(sizePolicy)
        self.rotateImageSpinBox.setMinimumSize(QtCore.QSize(104, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.rotateImageSpinBox.setFont(font)
        self.rotateImageSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.rotateImageSpinBox.setMinimum(0)
        self.rotateImageSpinBox.setMaximum(360)
        self.rotateImageSpinBox.setProperty("value", 0)
        self.rotateImageSpinBox.setObjectName("rotateImageSpinBox")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.rotateImageSpinBox)
        self.sendSmsLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.sendSmsLabel.setObjectName("sendSmsLabel")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.sendSmsLabel)
        self.sendSmsCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sendSmsCheckBox.setFont(font)
        self.sendSmsCheckBox.setObjectName("sendSmsCheckBox")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.sendSmsCheckBox)
        self.ratioCarBikeLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.ratioCarBikeLabel.setObjectName("ratioCarBikeLabel")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.ratioCarBikeLabel)
        self.ratioCarBikeSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ratioCarBikeSpinBox.sizePolicy().hasHeightForWidth())
        self.ratioCarBikeSpinBox.setSizePolicy(sizePolicy)
        self.ratioCarBikeSpinBox.setMinimumSize(QtCore.QSize(104, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.ratioCarBikeSpinBox.setFont(font)
        self.ratioCarBikeSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.ratioCarBikeSpinBox.setMinimum(1)
        self.ratioCarBikeSpinBox.setMaximum(9999)
        self.ratioCarBikeSpinBox.setObjectName("ratioCarBikeSpinBox")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.ratioCarBikeSpinBox)
        self.ratioPriorityBikeLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.ratioPriorityBikeLabel.setObjectName("ratioPriorityBikeLabel")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.ratioPriorityBikeLabel)
        self.ratioPriorityBikeSpinBox = QtWidgets.QSpinBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ratioPriorityBikeSpinBox.sizePolicy().hasHeightForWidth())
        self.ratioPriorityBikeSpinBox.setSizePolicy(sizePolicy)
        self.ratioPriorityBikeSpinBox.setMinimumSize(QtCore.QSize(104, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.ratioPriorityBikeSpinBox.setFont(font)
        self.ratioPriorityBikeSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.ratioPriorityBikeSpinBox.setMinimum(1)
        self.ratioPriorityBikeSpinBox.setMaximum(9999)
        self.ratioPriorityBikeSpinBox.setObjectName("ratioPriorityBikeSpinBox")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.ratioPriorityBikeSpinBox)
        self.statusLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.statusLabel.setObjectName("statusLabel")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.statusLabel)
        self.statusWidget = QtWidgets.QWidget(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.statusWidget.sizePolicy().hasHeightForWidth())
        self.statusWidget.setSizePolicy(sizePolicy)
        self.statusWidget.setMinimumSize(QtCore.QSize(210, 30))
        self.statusWidget.setObjectName("statusWidget")
        self.minSpinBox = QtWidgets.QSpinBox(self.statusWidget)
        self.minSpinBox.setGeometry(QtCore.QRect(50, 0, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.minSpinBox.setFont(font)
        self.minSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.minSpinBox.setMinimum(2)
        self.minSpinBox.setMaximum(99999)
        self.minSpinBox.setObjectName("minSpinBox")
        self.maxSpinBox = QtWidgets.QSpinBox(self.statusWidget)
        self.maxSpinBox.setGeometry(QtCore.QRect(160, 0, 42, 22))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.maxSpinBox.setFont(font)
        self.maxSpinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.maxSpinBox.setMinimum(3)
        self.maxSpinBox.setMaximum(99999)
        self.maxSpinBox.setObjectName("maxSpinBox")
        self.label_2 = QtWidgets.QLabel(self.statusWidget)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 47, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.statusWidget)
        self.label_3.setGeometry(QtCore.QRect(110, 0, 47, 21))
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.statusWidget)
        self.sendEmailLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.sendEmailLabel.setObjectName("sendEmailLabel")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.sendEmailLabel)
        self.sendEmailCheckBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.sendEmailCheckBox.setFont(font)
        self.sendEmailCheckBox.setObjectName("sendEmailCheckBox")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.sendEmailCheckBox)
        self.verticalLayout.addLayout(self.formLayout)

        self.retranslateUi(config_Dialog)
        self.buttonBox.accepted.connect(config_Dialog.accept)
        self.buttonBox.rejected.connect(config_Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(config_Dialog)

        # # Init default value default 
        # self.processImageEveryXFrameSpinBox.setValue(50)
        # self.createNewThreadEveryXFrameSpinBox.setValue(30)
        # self.checkResultEveryXFrameSpinBox.setValue(100)
        # self.sendResultEveryXFrameSpinBox.setValue(250)
        # self.createThreadForXFrameFirstSpinBox.setValue(300)
        # self.rotateImageSpinBox.setValue(0)
        # self.sendSmsCheckBox.setChecked(True)

        # Init default value
        self.processImageEveryXFrameSpinBox.setValue(10)
        self.createNewThreadEveryXFrameSpinBox.setValue(6)
        self.checkResultEveryXFrameSpinBox.setValue(20)
        self.sendResultEveryXFrameSpinBox.setValue(50)
        self.createThreadForXFrameFirstSpinBox.setValue(60)
        self.rotateImageSpinBox.setValue(0)
        self.sendSmsCheckBox.setChecked(True)
        self.sendEmailCheckBox.setChecked(True)
        self.ratioCarBikeSpinBox.setValue(2)
        self.ratioPriorityBikeSpinBox.setValue(3)
        self.minSpinBox.setValue(5)
        self.maxSpinBox.setValue(10)
        

    def retranslateUi(self, config_Dialog):
        _translate = QtCore.QCoreApplication.translate
        config_Dialog.setWindowTitle(_translate("config_Dialog", "Dialog"))
        self.label.setToolTip(_translate("config_Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color:#00aaff;\">Config</span></p></body></html>"))
        self.label.setText(_translate("config_Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:22pt; color:#0055ff;\">Config</span></p></body></html>"))
        self.processImageEveryXFrameLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Process image every X frame</span></p></body></html>"))
        self.createNewThreadEveryXFrameLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Create new thread every X frame</span></p></body></html>"))
        self.checkResultEveryXFrameLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Check result every X frame</span></p></body></html>"))
        self.sendResultEveryXFrameLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Send SMS every X frame</span></p></body></html>"))
        self.createThreadForXFrameFirstLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Create thread for X frame first</span></p></body></html>"))
        self.rotateImageLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Rotate image</span></p></body></html>"))
        self.sendSmsLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Send SMS</span></p></body></html>"))
        self.ratioCarBikeLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Ratio Car/Bike</span></p></body></html>"))
        self.ratioPriorityBikeLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Ratio Priority/Bike</span></p></body></html>"))
        self.statusLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Status (</span><span style=\" font-size:14pt; font-weight:600; color:#55aaff;\">Low</span><span style=\" font-size:14pt; color:#55aaff;\"> &lt; Min &lt;= </span><span style=\" font-size:14pt; font-weight:600; color:#55aaff;\">Medium</span><span style=\" font-size:14pt; color:#55aaff;\"> &lt; Max &lt;= </span><span style=\" font-size:14pt; font-weight:600; color:#55aaff;\">Traffic Jam</span><span style=\" font-size:14pt; color:#55aaff;\">)</span></p></body></html>"))
        self.label_2.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Min:</span></p></body></html>"))
        self.label_3.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Max:</span></p></body></html>"))
        self.sendEmailLabel.setText(_translate("config_Dialog", "<html><head/><body><p><span style=\" font-size:14pt; color:#55aaff;\">Send Email</span></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    config_Dialog = QtWidgets.QDialog()
    ui = Ui_config_Dialog()
    ui.setupUi(config_Dialog)
    config_Dialog.show()
    sys.exit(app.exec_())