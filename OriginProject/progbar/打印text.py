import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import tensorflow as tf
import pandas as pd

class Stream(QObject):
    newText = pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()
    def flush(self):
         self.content = ''

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.initGui()
        sys.stdout = Stream(newText=self.onUpdateText)
        self.network()

    def initGui(self):
        self.layout = QVBoxLayout()
        self.btn1 = QPushButton('开始训练')
        self.btn1.clicked.connect(self.startTrain)
        self.consoleBox = QTextEdit(self, readOnly=True)
        self.layout.addWidget(self.btn1)
        self.layout.addWidget(self.consoleBox)
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.show()

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

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.consoleBox.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.consoleBox.setTextCursor(cursor)
        self.consoleBox.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    sys.exit(app.exec_())
