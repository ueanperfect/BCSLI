import sys
from PyQt5.QtWidgets import QApplication,QWidget,QMainWindow,QTableWidgetItem,QTabWidget,QMessageBox
import UI
import W1
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QAbstractItemView
import PyQt5.QtWidgets as Qtw
import tensorflow as tf
import pandas as pd
import PyQt5.QtCore as Qt
import cgitb
import mwindow
cgitb.enable( format = 'text')

class log_in(QWidget,W1.Ui_Form):
    def __init__(self):
        super(log_in, self).__init__()
        self.setupUi(self)
        self.connect()
    def connect(self):
        self.pushButton.clicked.connect(self.to_window_2)
        self.pushButton.clicked.connect(self.close)
    def to_window_2(self):
         self.window_2=mwindow.mwindow()
         self.window_2.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #w = mwindow()
    #w.show()
    w=log_in()
    w.show()
    sys.exit(app.exec_())