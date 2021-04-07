import sys
from PyQt5.QtWidgets import QApplication,QWidget,QMainWindow,QTableWidgetItem,QTabWidget,QMessageBox
import UI
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import tensorflow as tf
import pandas as pd
from PyQt5.QtWidgets import QAbstractItemView
import PyQt5.QtWidgets as Qtw
import tensorflow as tf
import pandas as pd
import PyQt5.QtCore as Qt
import cgitb
cgitb.enable( format = 'text')


class Stream(QObject):
    newText = pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()
    def flush(self):
         self.content = ''

class mwindow(QWidget, UI.Ui_Form):
    def __init__(self):
        super(mwindow, self).__init__()
        self.setupUi(self)
        self.connect_BP()
        self.tableWidget_Set_BP()
        sys.stdout = Stream(newText=self.onUpdateText)
        #self.layout.addWidget(self.textBrowser_trainprogress)

        ##定义变量
        #网络结构变量
        self.x_trainPath=0
        self.y_trainPath=0
        self.xVariablePath=0
        self.y_testPath=0
        self.historyPath=0
        self.history=0
        self.number = []
        self.activation=[]
        self.NetworkName=[]
        #网络初始化变量
        self.model=tf.keras.Sequential([])

    def onUpdateText(self, text):
        """Write console output to text widget."""
        cursor = self.textBrowser_trainprogress.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser_trainprogress.setTextCursor(cursor)
        self.textBrowser_trainprogress.ensureCursorVisible()

    ##执行网络读取函数
    # 读取自变量训练集地址函数
    def x_trainPath_BP(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_x_trainPath_BP.setText(str(path[0][0]))
            self.x_trainPath = path[0][0]
            #print(self.x_trainPath)
    # 读取因变量训练集地址函数
    def y_trainPath_BP(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_y_trainPath_BP.setText(str(path[0][0]))
            self.y_trainPath = path[0][0]
            #print(self.y_trainPath)
    '''
    ######################## 
    '''
    ##网络结构更改
    #网络结构增加函数
    def NetworkAdd_BP(self):
        row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row)
        NetworkNameList_BP = ['Dense']
        comboBox_NetworkName_BP = Qtw.QComboBox()
        comboBox_NetworkName_BP.addItems(NetworkNameList_BP)
        self.tableWidget.setCellWidget(row, 0, comboBox_NetworkName_BP)
        PointNumber_BP=QTableWidgetItem('1')
        self.tableWidget.setItem(row,1,PointNumber_BP)
        comboBoxList2 = ['sigmoid','relu','softmax','None']
        comboBox2 = Qtw.QComboBox()
        comboBox2.addItems(comboBoxList2)
        self.tableWidget.setCellWidget(row, 2, comboBox2)
    #网络层删除函数
    def NetworkDelete_BP(self):
        row_select = self.tableWidget.selectedItems()
        #print(len(row_select))
        if len(row_select) == 0:
            return
        id = row_select[0].text()
        #print("id: {}".format(id))
        row = row_select[0].row()
        self.tableWidget.removeRow(row)

    #执行训练网络操作
    def startTraining_BP(self):
        if self.lineEdit_Epoch_BP.text()=='':
            QMessageBox.warning(self, "标题", "请输入训练次数", QMessageBox.Yes, QMessageBox.Yes)
            return
        else:
            self.history=self.model.fit(self.x_train,self.y_train,epochs=int(self.lineEdit_Epoch_BP.text()),verbose=2)
            QMessageBox.warning(self, "标题", "已经训练完成！", QMessageBox.Yes, QMessageBox.Yes)
            return
    ##输出历史数据页的函数
    #读取测试集数据地址函数
    def PredictxVariablePath_BP(self):
        path=Qtw.QFileDialog.getOpenFileNames(None,'选取文件')
        if path[0]==[]:
            return
        else:
            self.lineEdit_PredictxVariablePath_BP.setText(path[0][0])
            self.xVariablePath=path[0][0]
    #选择预测数据到哪个文件夹的函数
    def PredictDataPath_BP(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        self.y_testPath = path + '/y_test.xlsx'
        self.lineEdit_PredictDataPath_BP.setText(path)
    #输出预测数据函数
    def ExportPredictData_BP(self):
        if self.xVariablePath==0:
            QMessageBox.warning(self, "标题", "你还没有选择要预测数据输入地址", QMessageBox.Yes ,QMessageBox.Yes)
            return
        if self.y_testPath==0:
            QMessageBox.warning(self, "标题", "你还没有选择要把数据输出到哪个文件夹", QMessageBox.Yes, QMessageBox.Yes)
            return
        if self.y_testPath=='/y_test.xlsx':
            QMessageBox.warning(self, "标题", "你还没有选择要把数据输出到哪个文件夹", QMessageBox.Yes, QMessageBox.Yes)
            return
        else:
            self.x_test=pd.read_excel(self.xVariablePath)
            self.y_test=pd.DataFrame(self.model.predict(self.x_test))
            self.y_test.to_excel(self.y_testPath)
            QMessageBox.warning(self, "标题", "已输出数据", QMessageBox.Yes, QMessageBox.Yes)
            return
    #选择导出历史数据到哪个文件夹的函数
    def ExportTrainHistoryData_BP(self):
        if self.history==0:
            QMessageBox.warning(self, "标题", "你还没训练数据", QMessageBox.Yes, QMessageBox.Yes)
            return
        if self.historyPath==0:
            QMessageBox.warning(self, "标题", "你还没有选择输出地址", QMessageBox.Yes, QMessageBox.Yes)
            return
        if self.historyPath=='/history.xlsx':
            QMessageBox.warning(self, "标题", "你还没有选择输出地址", QMessageBox.Yes, QMessageBox.Yes)
            return
        else:
            historydata = pd.DataFrame(self.history.history['loss'])
            historydata.to_excel(self.historyPath)
            QMessageBox.warning(self, "标题", "已输出数据", QMessageBox.Yes, QMessageBox.Yes)
    #导出历史数据函数
    def PredictHistoryPath_BP(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        self.historyPath=path+'/history.xlsx'
        self.lineEdit_PredictHistoryPath_BP.setText(path)
        print(self.historyPath)
    '''
    ######################## 
    '''
    ##四个下一步函数
    #第一步读取数据+跳转下一页
    def ReadData_BP(self):
        if self.x_trainPath==0:
            QMessageBox.warning(self, "标题", "你还没有选择自变量地址", QMessageBox.Yes ,QMessageBox.Yes)
            return
        if self.y_trainPath==0:
            QMessageBox.warning(self, "标题", "你还没有选择因变量地址", QMessageBox.Yes, QMessageBox.Yes)
            return
        if self.lineEdit_xVaribleNumber_BP.text()=='':
            QMessageBox.warning(self, "标题", "请输入自变量个数", QMessageBox.Yes, QMessageBox.Yes)
        else:
            ##数据集读取
            self.x_train=pd.read_excel(self.x_trainPath)
            self.y_train=pd.read_excel(self.y_trainPath)
            ##跳转下一页
            self.TabWidget_BPNetwork.widget(0).setVisible(False)
            self.TabWidget_BPNetwork.widget(1).setVisible(True)
            self.TabWidget_BPNetwork.setCurrentIndex(1)
            self.TabWidget_BPNetwork.setTabEnabled(0,False)
    #第二步建立网络
    def Network_BP(self):
        ##网路读取与装备
        row = self.tableWidget.rowCount()
        for i in range(row):
            self.NetworkName.append(self.tableWidget.cellWidget(i,0).currentText())
            self.activation.append(self.tableWidget.cellWidget(i,2).currentText())
            self.number.append(int(self.tableWidget.item(i,1).text()))
        self.model.add(tf.keras.layers.Input(int(self.lineEdit_xVaribleNumber_BP.text())))
        for i in range(len(self.number)):
            if self.NetworkName[i] == 'Dense':
                if self.activation[i]!='None':
                    self.model.add(tf.keras.layers.Dense(self.number[i],activation=self.activation[i]))
                else:
                    self.model.add(tf.keras.layers.Dense(self.number[i]))
        #print(self.model.summary())
        self.TabWidget_BPNetwork.widget(1).setVisible(False)
        self.TabWidget_BPNetwork.widget(2).setVisible(True)
        self.TabWidget_BPNetwork.setCurrentIndex(2)
        self.TabWidget_BPNetwork.setTabEnabled(1, False)
    #第三部包装优化器
    def Optimizer_BP(self):
        if self.lineEdit_LearningRate_BP.text()=='':
            return QMessageBox.warning(self, "标题", "请输入学习速率", QMessageBox.Yes, QMessageBox.Yes)
        else:
            self.model.compile(optimizer=tf.optimizers.Adam(float(self.lineEdit_LearningRate_BP.text())),loss=self.comboBox_lossFunction_BP.currentText())
            #print(self.comboBox_lossFunction_BP.currentText())
            self.TabWidget_BPNetwork.widget(2).setVisible(False)
            self.TabWidget_BPNetwork.widget(3).setVisible(True)
            self.TabWidget_BPNetwork.setCurrentIndex(3)
            self.TabWidget_BPNetwork.setTabEnabled(2, False)
    #第四部运行网络
    def Run_BP(self):
        self.TabWidget_BPNetwork.widget(3).setVisible(False)
        self.TabWidget_BPNetwork.widget(4).setVisible(True)
        self.TabWidget_BPNetwork.setCurrentIndex(4)
        self.TabWidget_BPNetwork.setTabEnabled(3, False)
    #第五部关闭窗口——默认曹函数

    '''
    ######################## 
    '''
    def connect_BP(self):
        #读取数据界面按钮连接
        self.PushButton_y_trainPath_BP.clicked.connect(self.y_trainPath_BP)
        self.PushButton_x_trainPath_BP.clicked.connect(self.x_trainPath_BP)
        ##网络装配界面按钮连接
        self.PushButton_NetworkAdd_BP.clicked.connect(self.NetworkAdd_BP)
        self.PushButton_NetworkDelete_BP.clicked.connect(self.NetworkDelete_BP)
        ##训练网络按钮
        self.pushButton_startTraining_BP.clicked.connect(self.startTraining_BP)
        ##预测页面按钮
        #导出预测数据操作
        self.pushButton_PredictxVariablePath_BP.clicked.connect(self.PredictxVariablePath_BP)
        self.pushButton_PredictDataPath_BP.clicked.connect(self.PredictDataPath_BP)
        self.PushButton_ExportPredictData_BP.clicked.connect(self.ExportPredictData_BP)
        #导出数据历史
        self.pushButton_PredictHistoryPath_BP.clicked.connect(self.PredictHistoryPath_BP)
        self.PushButton_ExportTrainHistoryData_BP.clicked.connect(self.ExportTrainHistoryData_BP)
        # 四个页面的按妞装配
        self.PushButton_Next_ReadData_BP.clicked.connect(self.ReadData_BP)
        self.PushButton_Next_Network_BP.clicked.connect(self.Network_BP)
        self.PushButton_Next_Optimizer_BP.clicked.connect(self.Optimizer_BP)
        self.PushButton_Next_Run_BP.clicked.connect(self.Run_BP)
    def tableWidget_Set_BP(self):#初始化网络表格
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = mwindow()
    w.show()
    sys.exit(app.exec_())