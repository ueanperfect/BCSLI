from PyQt5 import QtWidgets
import sys
from PyQt5.QtWidgets import QWidget ,QApplication
import UI_STS
import pandas as pd
import PyQt5.QtWidgets as Qtw
from PyQt5.QtWidgets import QApplication,QWidget,QMainWindow,QTableWidgetItem,QTabWidget,QMessageBox
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class STSWidget(QWidget,UI_STS.Ui_Form):
    def __init__(self):
        super(STSWidget, self).__init__()
        self.setupUi(self)
        self.connect()
        self.convDataPath = ''
        self.mainDataPath = ''
        self.seasonalModuleName=[]
        self.convData=0

    def mainData_Path(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_readMainData_STS.setText(str(path[0][0]))
            self.mainDataPath = path[0][0]
            self.mainTrainData = pd.read_excel(self.mainDataPath, header=None)

    # 读取因变量训练集地址函数
    def convData_Path(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_readConvData_STS.setText(str(path[0][0]))
            self.convDataPath = path[0][0]
            self.convTrainData=pd.read_excel(self.convDataPath, header=None)

    def show_data(self):
        row = self.mainTrainData.shape[0]
        col = self.mainTrainData.shape[1]
        self.xVariable_number=col
        self.tableWidget_mainDataTable_STS.setColumnCount(col)
        self.tableWidget_mainDataTable_STS.setHorizontalHeaderLabels(self.mainTrainData.iloc[0].values)
        for i in range(1, row):
            rowlist = self.mainTrainData.iloc[i].values
            row = self.tableWidget_mainDataTable_STS.rowCount()
            self.tableWidget_mainDataTable_STS.insertRow(row)
            for j in range(len(rowlist)):
                newItem = QTableWidgetItem(str(rowlist[j]))
                self.tableWidget_mainDataTable_STS.setItem(i - 1, j, newItem)
        row = self.convTrainData.shape[0]
        col = self.convTrainData.shape[1]
        self.tableWidget_2.setColumnCount(col)
        self.tableWidget_2.setHorizontalHeaderLabels(self.convTrainData.iloc[0].values)
        for i in range(1, row):
            rowlist = self.convTrainData.iloc[i].values
            row = self.tableWidget_2.rowCount()
            self.tableWidget_2.insertRow(row)
            for j in range(len(rowlist)):
                newItem = QTableWidgetItem(str(rowlist[j]))
                self.tableWidget_2.setItem(i - 1, j, newItem)

    #页面2


#####四大换页按钮
    def ReadData_STS(self):
        if self.mainDataPath=='':
            QMessageBox.warning(self, "标题", "请选择自变量地址", QMessageBox.Yes ,QMessageBox.Yes)
            return
        if self.convDataPath=='':
            QMessageBox.warning(self, "标题", "请输入协变量", QMessageBox.Yes, QMessageBox.Yes)
            return
        else:
            ##跳转下一页
            self.mainTrainData=pd.read_excel(self.mainDataPath)
            self.convTrainData=pd.read_excel(self.convDataPath)
            self.tabWidget_sts.widget(0).setVisible(False)
            self.tabWidget_sts.widget(1).setVisible(True)
            self.tabWidget_sts.setCurrentIndex(1)
            self.tabWidget_sts.setTabEnabled(0,False)

    def seasonalModule(self):
        ##网路读取与装备
        self.seasonal=[]
        self.seasonalStep=[]
        self.seasonalModuleName=[]
        row = self.tableWidget_seasonalModule_STS.rowCount()
        for i in range(row):
            self.seasonalModuleName.append(self.tableWidget.cellWidget(i, 0).currentText())
            self.seasonalStep.append(int(self.tableWidget.cellWidget(i, 2).currentText()))
            #self.seasonalStepSon.append(int(self.tableWidget.item(i, 1).text()))
        self.convNumber=self.convData.shape[1]
        self.conv_effct=[]
        for i in range(self.convNumber):
            self.conv_effct.append(tfp.sts.LinearRegression(
                design_matrix=tf.reshape(self.convData[i]-np.mean(self.convData[i]),(-1,1)
            )),name=str('conv'+i))
        self.seasonal_effect=[]
        for i in range(row):
            self.seasonal_effect.append(tfp.sts.Seasonal(num_seasons=self.seasonalStep[i],
                                                         observed_time_series=self.mainTrainData,
                                                         name=self.seasonalModuleName[i]))
        self.AR = tfp.sts.Autoregressive(order=1, observed_time_series=self.mainTrainData, name='autoregressive')
        modelList=[]
        for i in range(self.convNumber):
            modelList.append(self.conv_effct[i])
        for i in range(row):
            modelList.append(self.seasonal_effect[i])
        if self.checkBox_ARModuleIf_STS.isChecked():
            modelList.append(self.AR)
        self.model = tfp.sts.Sum(modelList,observed_time_series=self.mainTrainData)
        self.variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=self.model)

        ###建立协变量网络层
        # for i in range(len(self.number)):
        #     if self.NetworkName[i] == 'Dense':
        #         if self.activation[i] != 'None':
        #             self.model.add(tf.keras.layers.Dense(self.number[i], activation=self.activation[i]))
        #         else:
        #             self.model.add(tf.keras.layers.Dense(self.number[i]))
        # print(self.model.summary())
        self.tabWidget_sts.widget(1).setVisible(False)
        self.tabWidget_sts.widget(2).setVisible(True)
        self.tabWidget_sts.setCurrentIndex(2)
        self.tabWidget_sts.setTabEnabled(1, False)

    def connect(self):
        self.pushButton_readMainData_STS.clicked.connect(self.mainData_Path)
        self.pushButton_readConvData_STS.clicked.connect(self.convData_Path)
        self.pushButton_showData.clicked.connect(self.show_data)
        self.pushButton_next_dataRead_STS.clicked.connect(self.ReadData_STS)
        self.pushButton_next_moduleStructure_STS.clicked.connect(self.seasonalModule)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w=STSWidget()
    w.show()
    sys.exit(app.exec_())

