from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

"""模型训练"""

train_dir='/Users/liyueyan/Desktop/图片处理4.0/训练文件'
# GRADED FUNCTION: train_happy_sad_model


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.96):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    # YOUR CODE ENDS HERE
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(200, 200),
                                                    batch_size=128,class_mode='binary')

callbacks = myCallback()
history=model.fit_generator(train_generator,epochs=1,callbacks=[callbacks])


"""加载待验证的图片"""

pre_dir='/Users/liyueyan/Desktop/图片处理4.0/计算文件'

pre_datagen = ImageDataGenerator(rescale=1 / 255)
pre_generator = pre_datagen.flow_from_directory(
        pre_dir,
        target_size=(200,200),
        batch_size=1,
        class_mode='binary')

label=model.predict_generator(pre_generator)

def load_img(generator):
    img_data=[]
    img_name=[]
    for index in range(len(generator)):
        image, label = generator._get_batches_of_transformed_samples(np.array([index]))
        image_name = generator.filenames[index]
        img_data.append(image)
        img_name.append(image_name)
    img_data = np.array(img_data)
    img_data=img_data.reshape(len(generator),200,200,3)
    return img_data,img_name

img_data,img_name = load_img(pre_generator)

plt.imshow(img_data[86])
print(img_name[45])
y=model(img_data)
y=np.array(y)

def flieWrite(data,path):#data为想导出的数据，path为导出数据存放的地址
    import pandas as pd
    data=pd.DataFrame(data)
    data.to_excel(path)

import pandas as pd
img_name = pd.DataFrame(img_name)
y=pd.DataFrame(y)

flieWrite(y,'/Users/liyueyan/Desktop/y.xlsx')
flieWrite(img_name,'/Users/liyueyan/Desktop/name.xlsx')

