import sys
import PyQt5.QtWidgets as Qtw
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import UI_STS
from PyQt5.QtWidgets import QApplication, QWidget, QTableWidgetItem, QMessageBox



# 画图函数
def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
    import seaborn as sns
    from matplotlib import pylab as plt
    import numpy as np
    """Plot a forecast distribution against the 'true' time series."""
    colors = sns.color_palette()
    c1, c2 = colors[0], colors[1]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    num_steps = len(y)
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    ax.plot(x, y, lw=2, color=c1, label='ground truth')

    forecast_steps = np.arange(
        x[num_steps_train],
        x[num_steps_train] + num_steps_forecast,
        dtype=x.dtype)

    ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

    ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
            label='forecast')
    ax.fill_between(forecast_steps,
                    forecast_mean - 2 * forecast_scale,
                    forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

    ymin, ymax = min(
        np.min(forecast_samples), np.min(y)), max(
        np.max(forecast_samples), np.max(y))
    yrange = ymax - ymin
    ax.set_ylim([ymin - yrange * 0.1, ymax + yrange * 0.1])
    ax.set_title("{}".format(title))
    ax.legend()

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
        ax.xaxis.set_major_formatter(x_formatter)
        fig.autofmt_xdate()

    return fig, ax


class STSWidget(QWidget, UI_STS.Ui_Form):
    def __init__(self):
        super(STSWidget, self).__init__()
        self.setupUi(self)
        self.connect()
        self.convDataPath = ''
        self.mainDataPath = ''
        self.convTestDataPath = ''
        self.lossPath = ''
        self.predictionDataPath = ''
        self.seasonalModuleName = []
        self.convData = 0

    def mainData_Path_STS(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_readMainData_STS.setText(str(path[0][0]))
            self.mainDataPath = path[0][0]
            self.mainTrainData = pd.read_excel(self.mainDataPath, header=None)

    def convData_Path_STS(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_readConvData_STS.setText(str(path[0][0]))
            self.convDataPath = path[0][0]
            self.convTrainData = pd.read_excel(self.convDataPath, header=None)

    def testConvData_Path_STS(self):
        path = Qtw.QFileDialog.getOpenFileNames(None, "选取文件夹")
        if path[0] == []:
            return
        else:
            self.lineEdit_testConvData.setText(str(path[0][0]))
            self.convTestDataPath = path[0][0]
            self.convTestData = pd.read_excel(
                self.convTestDataPath, header=None)

    def show_data_STS(self):
        if self.mainDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请选择自变量地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        if self.convDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请输入协变量",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        if self.convTestDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请选择协变量测试集地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        row = self.mainTrainData.shape[0]
        col = self.mainTrainData.shape[1]
        self.xVariable_number = col
        self.tableWidget_mainDataTable_STS.setColumnCount(col)
        self.tableWidget_mainDataTable_STS.setHorizontalHeaderLabels(
            self.mainTrainData.iloc[0].values)
        for i in range(1, row):
            rowlist = self.mainTrainData.iloc[i].values
            row = self.tableWidget_mainDataTable_STS.rowCount()
            self.tableWidget_mainDataTable_STS.insertRow(row)
            for j in range(len(rowlist)):
                newItem = QTableWidgetItem(str(rowlist[j]))
                self.tableWidget_mainDataTable_STS.setItem(i - 1, j, newItem)
        row = self.convTrainData.shape[0]
        col = self.convTrainData.shape[1]
        self.tableWidget_convDataTable_STS.setColumnCount(col)
        self.tableWidget_convDataTable_STS.setHorizontalHeaderLabels(
            self.convTrainData.iloc[0].values)
        for i in range(1, row):
            rowlist = self.convTrainData.iloc[i].values
            row = self.tableWidget_convDataTable_STS.rowCount()
            self.tableWidget_convDataTable_STS.insertRow(row)
            for j in range(len(rowlist)):
                newItem = QTableWidgetItem(str(rowlist[j]))
                self.tableWidget_convDataTable_STS.setItem(i - 1, j, newItem)
        row = self.convTestData.shape[0]
        col = self.convTestData.shape[1]
        self.tableWidget_testCoverData_STS.setColumnCount(col)
        self.tableWidget_testCoverData_STS.setHorizontalHeaderLabels(
            self.convTestData.iloc[0].values)
        for i in range(1, row):
            rowlist = self.convTestData.iloc[i].values
            row = self.tableWidget_testCoverData_STS.rowCount()
            self.tableWidget_testCoverData_STS.insertRow(row)
            for j in range(len(rowlist)):
                newItem = QTableWidgetItem(str(rowlist[j]))
                self.tableWidget_testCoverData_STS.setItem(i - 1, j, newItem)

    def addModule_STS(self):
        row = self.tableWidget_seasonalModule_STS.rowCount()
        self.tableWidget_seasonalModule_STS.insertRow(row)

    def deleteModule_STS(self):
        index = self.tableWidget_seasonalModule_STS.selectedRanges()  # 1
        index = index[0].bottomRow()
        self.tableWidget_seasonalModule_STS.removeRow(index)

    def printModel_STS(self):
        self.textBrowser_showModel_STS.setText(str(self.modelList))

    def trainModel_STS(self):
        if self.lineEdit.text() == '':
            QMessageBox.warning(
                self,
                "标题",
                "请迭代次数",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        steps = int(self.lineEdit.text())
        optimizer = tf.optimizers.Adam(learning_rate=0.1)
        self.elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.model.joint_log_prob(
                observed_time_series=self.mainTrainData),
            surrogate_posterior=self.variational_posteriors,
            optimizer=optimizer,
            num_steps=steps)
        self.elbo_loss_curve = np.array(self.elbo_loss_curve)
        ##画画
        q_samples_demand_ = self.variational_posteriors.sample(50)
        forecast_dist = tfp.sts.forecast(model=self.model, observed_time_series=self.mainTrainData,
                                         parameter_samples=q_samples_demand_,
                                         num_steps_forecast=len(self.convTestData))
        num_samples = 40
        (
            self.forecast_mean,
            self.forecast_scale,
            self.forecast_samples
        ) = (
            forecast_dist.mean().numpy()[..., 0],
            forecast_dist.stddev().numpy()[..., 0],
            forecast_dist.sample(num_samples).numpy()[..., 0]
        )

    def lossPath(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        if path == []:
            return
        else:
            self.lineEdit_lossPath.setText(str(path))
            self.lossPath = str(path)

    def predictionDataPath(self):
        path = Qtw.QFileDialog.getExistingDirectory(None, "选取文件夹")
        if path == []:
            return
        else:
            self.lineEdit_predictionDataPath.setText(str(path))
            self.predictionDataPath = str(path)

    def exportLossData(self):
        if self.lossPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "你还没有选择地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        loss = pd.DataFrame(self.elbo_loss_curve)
        loss.to_excel(self.lossPath + '/loss.xlsx')

    def exportPredictionData(self):
        if self.predictionDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "你还没有选择地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        forecastMean = pd.DataFrame(self.forecast_mean)
        forecaststd = pd.DataFrame(self.forecast_scale)
        forecastMean.to_excel(self.predictionDataPath + '/mean.xlsx')
        forecaststd.to_excel(self.predictionDataPath + '/std.xlsx')

    # 四大换页按钮
    def ReadData_STS(self):
        if self.mainDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请选择自变量地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        if self.convDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请输入协变量",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        if self.convTestDataPath == '':
            QMessageBox.warning(
                self,
                "标题",
                "请选择协变量测试集地址",
                QMessageBox.Yes,
                QMessageBox.Yes)
            return
        else:
            # 跳转下一页
            self.mainTrainData = np.array(pd.read_excel(self.mainDataPath))
            self.convTrainData = np.array(pd.read_excel(self.convDataPath))
            self.convTestData = np.array(pd.read_excel(self.convTestDataPath))
            self.convData = np.concatenate(
                (self.convTrainData, self.convTestData), axis=0)
            ##设置可视化
            self.tabWidget_sts.widget(0).setVisible(False)
            self.tabWidget_sts.widget(1).setVisible(True)
            self.tabWidget_sts.setCurrentIndex(1)
            self.tabWidget_sts.setTabEnabled(0, False)

    def seasonalModule_STS(self):
        # 网路读取与装备
        ## 设置读取季节性模块
        self.seasonalStep = []
        self.seasonalModuleName = []
        self.row = self.tableWidget_seasonalModule_STS.rowCount()
        for i in range(self.row):
            self.seasonalModuleName.append(
                str(self.tableWidget_seasonalModule_STS.item(i, 0).text()))
            self.seasonalStep.append(
                int(self.tableWidget_seasonalModule_STS.item(i, 1).text()))

        self.seasonal_effect = []
        for i in range(self.row):
            self.seasonal_effect.append(tfp.sts.Seasonal(num_seasons=self.seasonalStep[i],
                                                         observed_time_series=self.mainTrainData,
                                                         name=self.seasonalModuleName[i]))
        ##设置conv性模块
        self.convNumber = self.convData.shape[1]
        self.conv_effct = []
        for i in range(self.convNumber):
            self.conv_effct.append(tfp.sts.LinearRegression(
                design_matrix=tf.reshape(self.convData[:, i] - np.mean(self.convData[:, i]), (-1, 1)
                                         )))
        ##设置自回归模块
        self.AR = tfp.sts.Autoregressive(
            order=1,
            observed_time_series=self.mainTrainData,
            name='autoregressive')

        ##设置模块集合
        self.modelList = []
        for i in range(self.convNumber):
            self.modelList.append(self.conv_effct[i])
        for i in range(self.row):
            self.modelList.append(self.seasonal_effect[i])
        if self.checkBox_ARModuleIf_STS.isChecked():
            self.modelList.append(self.AR)

        self.model = tfp.sts.Sum(
            self.modelList, observed_time_series=self.mainTrainData)
        self.variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
            model=self.model)

        ##页面设置
        self.tabWidget_sts.widget(1).setVisible(False)
        self.tabWidget_sts.widget(2).setVisible(True)
        self.tabWidget_sts.setCurrentIndex(2)
        self.tabWidget_sts.setTabEnabled(1, False)

    def modelPrint_STS(self):
        ##页面设置
        self.tabWidget_sts.widget(2).setVisible(False)
        self.tabWidget_sts.widget(3).setVisible(True)
        self.tabWidget_sts.setCurrentIndex(3)
        self.tabWidget_sts.setTabEnabled(2, False)

    def connect(self):
        self.pushButton_readMainData_STS.clicked.connect(self.mainData_Path_STS)
        self.pushButton_readConvData_STS.clicked.connect(self.convData_Path_STS)
        self.pushButton_testConvData.clicked.connect(self.testConvData_Path_STS)
        self.pushButton_showData.clicked.connect(self.show_data_STS)
        self.pushButton_next_dataRead_STS.clicked.connect(self.ReadData_STS)
        self.pushButton_next_moduleStructure_STS.clicked.connect(
            self.seasonalModule_STS)
        self.pushButton_addModule_STS.clicked.connect(self.addModule_STS)
        self.pushButton_deleteModule_STS.clicked.connect(self.deleteModule_STS)
        self.pushButton_next_modelPrint_STS.clicked.connect(
            self.modelPrint_STS)
        self.pushButton_trainModel_STS.clicked.connect(self.trainModel_STS)
        self.pushButton_modelPrint_STS.clicked.connect(self.printModel_STS)
        self.pushButton_lossPath.clicked.connect(self.lossPath)
        self.pushButton_predictionDataPath.clicked.connect(self.predictionDataPath)
        self.pushButton_exportPredictionData.clicked.connect(self.exportPredictionData)
        self.pushButton_exportLossData.clicked.connect(self.exportLossData)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = STSWidget()
    w.show()
    sys.exit(app.exec_())
