import sys
import PyQt5.QtWidgets as Qtw
import numpy as np
import pandas as pd
import tensorflow as tf
import UI_CNN
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidgetItem, QMessageBox
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class STSWidget(QWidget, UI_CNN.Ui_Form):
    def __init__(self):
        super(STSWidget, self).__init__()
        self.setupUi(self)
        self.connect()

    def picPath_CNN(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        train_datagen = ImageDataGenerator(rescale=1 / 255)

        self.train_generator = train_datagen.flow_from_directory(path, target_size=(200, 200),
                                                            batch_size=128, class_mode='binary')
        self.lineEdit_picPath_CNN.setText(path)

    def Add_CNN(self):
        row = self.tableWidget_modelStructor_CNN.rowCount()
        self.tableWidget_modelStructor_CNN.insertRow(row)

        NetworkNameList_BP = ['Dense','Conv','MaxPooling2D','Flatten']
        comboBox_NetworkName_BP = Qtw.QComboBox()
        comboBox_NetworkName_BP.addItems(NetworkNameList_BP)
        self.tableWidget_modelStructor_CNN.setCellWidget(row, 0, comboBox_NetworkName_BP)

        PointNumber_BP = QTableWidgetItem('0')
        self.tableWidget_modelStructor_CNN.setItem(row, 1, PointNumber_BP)

        PointNumber_BP = QTableWidgetItem('0')
        self.tableWidget_modelStructor_CNN.setItem(row, 2, PointNumber_BP)

        comboBoxList2 = ['sigmoid', 'relu', 'softmax', 'None']
        comboBox2 = Qtw.QComboBox()
        comboBox2.addItems(comboBoxList2)
        self.tableWidget_modelStructor_CNN.setCellWidget(row, 3, comboBox2)

    def delete_CNN(self):
        row_select = self.tableWidget_modelStructor_CNN.selectedItems()
        # print(len(row_select))
        if len(row_select) == 0:
            return
        id = row_select[0].text()
        # print("id: {}".format(id))
        row = row_select[0].row()
        self.tableWidget_modelStructor_CNN.removeRow(row)

    def beginTrain_CNN(self):
        self.model.compile(optimizer=tf.optimizers.RMSprop(lr=float(self.lineEdit_lr.text())),loss='binary_crossentropy', metrics=['acc'])
        self.loss=self.model.fit(self.train_generator,epochs=int(self.lineEdit_Step_CNN.text()))

    def picTestPath_CNN(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        test_datagen = ImageDataGenerator(rescale=1 / 255)

        self.test_generator = test_datagen.flow_from_directory(path, target_size=(200, 200),
                                                                 batch_size=128, class_mode='binary')
        self.lineEdit_picTestPath_CNN.setText(path)

    def exportPredictLabel(self):
        def load_img(generator):
            img_data = []
            img_name = []
            for index in range(len(generator)):
                image, label = generator._get_batches_of_transformed_samples(np.array([index]))
                image_name = generator.filenames[index]
                img_data.append(image)
                img_name.append(image_name)
            img_data = np.array(img_data)
            img_data = img_data.reshape(len(generator), 200, 200, 3)
            return img_data, img_name

        img_data, img_name = load_img(self.test_generator)
        predictLabel=self.model(img_data)
        self.predictLabel=pd.DataFrame(np.array(predictLabel))
        self.img_name = pd.DataFrame(img_name)
        path = Qtw.QFileDialog.getSaveFileName(None, "选取文件夹","label.xlsx")
        self.predictLabel.to_excel(path[0])
        self.img_name.to_excel(path[0])
        return

    def exportLoss(self):
        loss=pd.DataFrame(self.loss['loss'])
        path=Qtw.QFileDialog.getSaveFileName(None, "选取文件夹","loss.xlsx")
        loss.to_excel(path[0])

    def Next_modelBulid_CNN(self):
        self.model = tf.keras.Sequential([])
        # 网路读取与装备
        self.NetworkName = []
        self.number1 = []
        self.number2 = []
        self.activation = []
        row = self.tableWidget_modelStructor_CNN.rowCount()
        for i in range(row):
            self.NetworkName.append(
                self.tableWidget_modelStructor_CNN.cellWidget(
                    i, 0).currentText())
            self.number1.append(int(self.tableWidget_modelStructor_CNN.item(i, 1).text()))
            self.number2.append(int(self.tableWidget_modelStructor_CNN.item(i, 2).text()))
            self.activation.append(
                self.tableWidget_modelStructor_CNN.cellWidget(
                    i, 3).currentText())

        # self.model.add(tf.keras.layers.Input(int(self.xVariable_number)))
        for i in range(len(self.number1)):
            if self.NetworkName[i] == 'Conv':
                if i == 0:
                    self.model.add(tf.keras.layers.Conv2D(
                        self.number1[i], (self.number2[i], self.number2[i]), activation=self.activation[i],
                        input_shape=(200, 200, 3)))
                else:
                    self.model.add(tf.keras.layers.Conv2D(
                        self.number1[i], (self.number2[i], self.number2[i]), activation=self.activation[i]))

            if self.NetworkName[i] == 'MaxPooling2D':
                self.model.add(tf.keras.layers.MaxPooling2D(self.number2[i], self.number2[i]))

            if self.NetworkName[i] == 'Flatten':
                self.model.add(tf.keras.layers.Flatten())

            if self.NetworkName[i] == 'Dropout':
                self.model.add(tf.keras.layers.Dropout(self.number1[i]))

            if self.NetworkName[i] == 'Dense':
                if self.activation[i] != 'None':
                    self.model.add(
                        tf.keras.layers.Dense(
                            self.number1[i],
                            activation=self.activation[i]))
                else:
                    self.model.add(tf.keras.layers.Dense(self.number1[i]))

        print(self.model.summary())
        self.tabWidget_CNN.widget(1).setVisible(False)
        self.tabWidget_CNN.widget(2).setVisible(True)
        self.tabWidget_CNN.setCurrentIndex(2)
        self.tabWidget_CNN.setTabEnabled(1, False)

    def Next_dataRead_CNN(self):
        if self.lineEdit_picPath_CNN.text() == '':
            QMessageBox.warning(
                self,
                "标题",
                "你还没有选择图片地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return

        self.tabWidget_CNN.widget(0).setVisible(False)
        self.tabWidget_CNN.widget(1).setVisible(True)
        self.tabWidget_CNN.setCurrentIndex(1)
        self.tabWidget_CNN.setTabEnabled(0, False)

    def Next_modelTrain_CNN(self):
        self.tabWidget_CNN.widget(2).setVisible(False)
        self.tabWidget_CNN.widget(3).setVisible(True)
        self.tabWidget_CNN.setCurrentIndex(3)
        self.tabWidget_CNN.setTabEnabled(2, False)

    def connect(self):
        self.pushButton_picPath_CNN.clicked.connect(self.picPath_CNN)
        self.pushButton_Next_dataRead_CNN.clicked.connect(self.Next_dataRead_CNN)
        self.pushButton_Add_CNN.clicked.connect(self.Add_CNN)
        self.pushButton_delete_CNN.clicked.connect(self.delete_CNN)
        self.pushButton_Next_modelBulid_CNN.clicked.connect(self.Next_modelBulid_CNN)
        self.pushButton_beginTrain_CNN.clicked.connect(self.beginTrain_CNN)
        self.pushButton_picTestPath_CNN.clicked.connect(self.picTestPath_CNN)
        self.pushButton_exportPredictLabel.clicked.connect(self.exportPredictLabel)
        self.pushButton_exportLoss.clicked.connect(self.exportLoss)
        self.pushButton_Next_modelTrain_CNN.clicked.connect(self.Next_modelTrain_CNN)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = STSWidget()
    w.show()
    sys.exit(app.exec_())