from PyQt5.QtWidgets import QMainWindow, QProgressBar, QApplication, QLabel, \
    QStatusBar, QPushButton

import sys


class SampleBar(QMainWindow):
    """Main Application"""

    def __init__(self, parent=None):
        # print('Starting the main Application')
        super(SampleBar, self).__init__(parent)
        self.initUI()

    def initUI(self):
        # Pre Params:
        self.setMinimumSize(800, 600)

        # File Menus & Status Bar:
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet('QStatusBar::item {border: none;}')
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('准备')
        self.progressBar = QProgressBar()
        self.pushbutton = QPushButton("点这里")
        self.label = QLabel()
        self.label2 = QLabel()
        self.label.setText("开始计算  ")
        self.label2.setText("正在计算： ")

        # self.statusBar.addWidget(self.label, 0)
        self.statusBar.addPermanentWidget(self.label, stretch=2)
        self.statusBar.addPermanentWidget(self.label2, stretch=0)
        self.statusBar.addPermanentWidget(self.progressBar, stretch=4)
        # self.statusBar().addWidget(self.progressBar)

        # This is simply to show the bar
        # self.progressBar.setGeometry(0, 0, 100, 5)
        self.progressBar.setRange(0, 500)  # 设置进度条的范围
        self.progressBar.setValue(20)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main2 = SampleBar()
    main2.show()
    sys.exit(app.exec_())

