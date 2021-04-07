##这个包是作为使用tensorflow的一些操作时所需要的包的打包。
##由于以前的使用频繁的查询网页，我制作了这些包来进行原始代码的简单运算。

def datasetToTensor(data,tagetName):#制作数据集 tagetName的数据为预测目标
    import tensorflow as tf
    target=data.pop(tagetName)
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    return dataset

def datasetComplize(x_train,y_train,x_test,y_test,SHUFFLE_VALUE,BATH_VALUE):##注意，x，y的数据结构，并且还要注意他么都是数组的数据类型
    import tensorflow as tf
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_db = train_db.shuffle(SHUFFLE_VALUE)
    test_db = test_db.shuffle(SHUFFLE_VALUE)
    train_db = train_db.batch(BATH_VALUE)
    test_db = test_db.batch(BATH_VALUE)
    return train_db,test_db

def dataSlices(x,y,TEST_SIZE,RANDOM_STATE):#切分数据使用
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_train = tf.cast(train_x, dtype=tf.float32)
    y_train = tf.cast(train_y, dtype=tf.float32)
    x_test = tf.cast(test_x, dtype=tf.float32)
    y_test = tf.cast(test_y, dtype=tf.float32)
    return x_train,y_train,x_test,y_test

def dataTensorboard(path):#记得在终端通过tensorboard --logdir=path的网页版本
    import datetime
    import tensorflow as tf
    log_dir = path+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)  # 定义TensorBoard对象
    return tensorboard_callback


