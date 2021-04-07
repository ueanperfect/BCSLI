from PyQt5.QtWidgets import QApplication ,QWidget
import UI
import sys
from PyQt5 import QtWidgets
import pandas as pd

class Widge(QWidget,UI.Ui_Form):
    def __init__(self):
        super(Widge, self).__init__()
        self.setupUi(self)
        self.connect()

    def readExcel(self):
        path = QtWidgets.QFileDialog.getOpenFileNames(None, "选取文件")
        self.excelPath=path[0][0]
        self.openExcel()
        print(self.data)

    def openExcel(self):
        self.data=pd.read_excel(self.excelPath)



    def connect(self):
        self.pushButton.clicked.connect(self.readExcel)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widge()
    w.show()
    sys.exit(app.exec_())