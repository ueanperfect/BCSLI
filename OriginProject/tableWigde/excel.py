import sys
from PyQt5.QtWidgets import *

from tableWigde import readExcel


class main_widget(QWidget):  # 继承自 QWidget类
    def __init__(self):
        super().__init__()
        self.initUI()  # 创建窗口

    def initUI(self):
        # 在此处添加 窗口控件
        self.setGeometry(200, 300, 1000, 600)  # 屏幕上坐标（x, y）， 和 窗口大小(宽，高)
        self.setWindowTitle("电子BOM表辅助工具")
        hbox = QHBoxLayout(self)  # 创建布局，可以让控件随着窗口的改变而改变
        self.onewidget = QFrame()  # 创建一个QFrame窗口。Qwidget也可以
        self.tableWidget3 = QTableWidget(1, 7)  # 创建一个表格tablewidget
        # 创建表头
        self.tableWidget3.setHorizontalHeaderLabels(
            ['品名', 'SR P/N', 'MPN', '规格型号描述', '品牌（MFG）', 'RD窗口', '备注'])
        # 禁止编辑
        self.tableWidget3.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 添加tableWidget去hbox中
        hbox.addWidget(self.tableWidget3)
        self.setLayout(hbox)
        readExcel.read_excel(self.tableWidget3)
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # path = '/Users/sr00117/Desktop/GUI/images/cat_597px_1221818_easyicon.net.png'
    # app.setWindowIcon(QIcon(path))  # MAC 下 程序图标是显示在程序坞中的， 切记；
    window = main_widget()
    sys.exit(app.exec_())
