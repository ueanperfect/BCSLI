import xlrd
from PyQt5.QtWidgets import *
import pandas as pd

def read_excel(tableWidget3):
    # 打开文件
    workbook = pd.read_excel('2018.xlsx')
    # 获取所有sheet
    #sheet2_name = workbook.sheet_names()[0]
    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_index(0) # sheet索引从0开始
    cols = sheet1.col_values(1)  # 获取第三列内容 品名
    # 获取整行和整列的值（数组）
    for i in range(4,len(cols)):
        rowslist = sheet1.row_values(i) # 获取excel每行内容
        for j in range(len(rowslist)):
            #在tablewidget中添加行
            row = tableWidget3.rowCount()
            tableWidget3.insertRow(row)
            #把数据写入tablewidget中
            newItem = QTableWidgetItem(rowslist[j])
            tableWidget3.setItem(i-4, j-1, newItem)
