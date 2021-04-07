from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QTimer, Qt
import sys
import time
import random


class Ui_Test_Transparent(object):
    def setupUi(self, Test_Transparent):
        Test_Transparent.setObjectName("Test_Transparent")
        Test_Transparent.resize(850, 620)
        Test_Transparent.setMinimumSize(QtCore.QSize(850, 620))
        Test_Transparent.setMaximumSize(QtCore.QSize(850, 620))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./img/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Test_Transparent.setWindowIcon(icon)
        Test_Transparent.setWindowOpacity(1)
        # Test_Transparent.setWindowFlags(Qt.FramelessWindowHint)
        self.label = QtWidgets.QLabel(Test_Transparent)
        self.label.setEnabled(False)
        self.label.setGeometry(QtCore.QRect(0, 0, 851, 621))
        self.label.setStyleSheet("")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("./img/x.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Test_Transparent)
        self.label_2.setGeometry(QtCore.QRect(155, 130, 551, 341))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.retranslateUi(Test_Transparent)
        QtCore.QMetaObject.connectSlotsByName(Test_Transparent)

    def retranslateUi(self, Test_Transparent):
        _translate = QtCore.QCoreApplication.translate
        Test_Transparent.setWindowTitle(_translate("Test_Transparent", "心灵毒鸡汤"))
        self.timer = QTimer(Test_Transparent)
        self.timer.start(50)
        self.timer.timeout.connect(self.show)

        # 语录列表
        a1 = "我感觉我也累了，不折腾了。可能我不配幸福吧，我认命了。"
        a2 = "我以为我能逗你笑你就会喜欢我，可我却输给了让你哭的人。"
        a3 = "若能避开猛烈的欢喜，自然不会有悲痛袭来。"
        a4 = "大学一转眼四年就过去了。"
        self.sentense = [a1, a2, a3, a4]
        self.label_2.setText(random.choice(self.sentense))

        # counter用于根据计时器更新窗体透明度
        self.counter = 1
        # degree用于控制透明度的增减性（“显示-透明-显示-透明”为一个循环周期）
        self.degree = -0.03

    def show(self):
        if self.counter >= 1:
            time.sleep(4)
            self.counter -= 0.01
            self.degree *= -1
        else:
            window.setWindowOpacity(self.counter)
            self.counter -= self.degree
        if self.counter < 0:
            self.label_2.setText(random.choice(self.sentense))
            self.degree *= -1
            self.counter = 0.01


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QDialog()
    ui = Ui_Test_Transparent()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())

