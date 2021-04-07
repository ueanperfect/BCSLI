import tensorflow as tf
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

class redirect:
    content = ''
    def write(self,str):
        self.content += str
    def flush(self):
        self.content = ''

r = redirect()
sys.stdout = r

# for i in range(10):
#     print(i)


pathx='/Users/liyueyan/Desktop/x_train.xlsx'
pathy='/Users/liyueyan/Desktop/y_train.xlsx'

x_train=pd.read_excel(pathx)
y_train=pd.read_excel(pathy)
# Callbacks
model=tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.optimizers.RMSprop(0.001),loss='mean_squared_error')
history=model.fit(x_train,y_train,epochs=50,verbose=2)

historydata=pd.DataFrame(history.history['loss'])