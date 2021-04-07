from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import UI
import sys
import tensorflow as tf
import pandas as pd

class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()

class Widge(QWidget,UI.Ui_Form):

    def __init__(self):
        super(Widge, self).__init__()
        self.setupUi(self)
        self.connect()
        self.network()

    def network(self):
        self.pathx = '/Users/liyueyan/Desktop/x_train.xlsx'
        self.pathy = '/Users/liyueyan/Desktop/y_train.xlsx'

        self.x_train = pd.read_excel(self.pathx)
        self.y_train = pd.read_excel(self.pathy)
        # Callbacks
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(5),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer=tf.optimizers.RMSprop(0.001), loss='mean_squared_error')
        self.model.summary()

    def startTrain(self):
        self.model.fit(self.x_train, self.y_train, epochs=20,verbose=2)
        return

    def connect(self):
        self.pushButton.clicked.connect(self.startTrain)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widge()
    w.show()
    sys.exit(app.exec_())

