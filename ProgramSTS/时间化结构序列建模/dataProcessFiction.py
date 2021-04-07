##所有的函数基本都基于pandas库的函数
##这些函数起到一个简化数据处理的功能

def twoToThreeD(data_x,data_y):#矩阵转化为三维张量
    import numpy as np
    data_x=np.reshape(data_x,(-1,len(data_y),1))
    return data_x

##写入数据函数
def dataRead(path):#path文件为数据表格excel的常用数据地址
    import pandas as pd
    data=pd.read_excel(path)
    return data

##输出数据
def flieWrite(data,path):#data为想导出的数据，path为导出数据存放的地址
    import pandas as pd
    data=pd.DataFrame(data)
    data.to_excel(path)

#检查空数据组
def checkNullData(data):#data为原始数据，这个函数的作用是检查空缺数据数量。
    list=data.columns.values
    for i in list:
        missNumber=data[i].isnull().value_counts()
        print(missNumber)#这个函数的一打功能是定空缺数据所在的位置

##删除空数据组
def deleteNullData(data):#data为原始数据，这个函数返回一个删除掉空缺数据的数据集。
    data=data.dropna()
    return data

##检查重复数据组
def checkDuplicateValue(data):#检查重复的数据集的个数以及定位
    duplicateDataCout=data.duplicated()
    print(duplicateDataCout)

#删除重复数据组
def deleteDuplicateData(data):#删除data中的空白数据集
    data=data.drop_duplicates()
    return data

##一个函数处理好所有的数据
def oneProesse(data):#一步清晰data中的文件项目
    data=deleteNullData(data)
    data=deleteDuplicateData(data)
    return(data)

##一步到位
def oneStep(path):#从读取地址到最后一步清洗，一步到位
    data=dataRead(path)
    data=oneProesse(data)
    return data
##热码处理

def oneHotProcess(data,label):#将序列号为热码的序列号
    import pandas as pd
    var_to_encode = label
    data = pd.get_dummies(data, columns=var_to_encode)
    return data

##object转化为离散数值
def dataObjectConvert(data,label):#将数据中这个标签数据整化为1，2，3，4，5序列号
    import pandas as pd
    data[label]=pd.Categorical(data[label])
    data[label] = data.thal.cat.codes

##将数据分离
def dataSeparate(data,label):#将数据中这个标签的数据删除。
    import numpy as np
    import pandas as pd
    y = data[label]
    y = pd.DataFrame(np.array(y).reshape(-1,1))
    x=data.drop([label],axis=1)
    return x,y

def fillData(data,wrongname):#将有问题的数据替换成上一个
    for i in range(len(data)):
        if data[i] == wrongname:
            data[i] = data[i - 1]
    return data



