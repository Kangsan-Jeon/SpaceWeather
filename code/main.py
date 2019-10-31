#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
import PIL
import keras

np.random.seed(3)
nb_class = 4



# 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.1)

train_generator = train_datagen.flow_from_directory('JKS/power/train',
                                                   target_size=(224,224),
                                                   batch_size=32, class_mode='categorical',
                                                   subset = 'training')

val_generator = train_datagen.flow_from_directory('JKS/power/train',
                                                   target_size=(224,224),
                                                   batch_size=32, class_mode='categorical',
                                                   subset = 'validation')


#validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

#validation_generator = validation_datagen.flow_from_directory('FlareForecast/power/train',
#                                                   target_size=(224,224),
#                                                   batch_size=32, class_mode='categorical')


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = train_datagen.flow_from_directory('JKS/power/test',
                                                   target_size=(224,224),
                                                   batch_size=32, class_mode='categorical')

# 모델 구성하기 VGG16
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_class, activation='softmax'))


# 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

e_step = len(train_generator)
v_step = len(val_generator)
t_step = len(test_generator)

print(e_step, v_step, t_step)

# 모델 학습시키기
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=20) # 조기종료 콜백함수 정의

# 모델 학습시키기
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
                                     write_graph=True, write_images=True)

# hist = model.fit(X_train, Y_train, epochs=20, batch_size=128,
#           validation_data=(X_val, Y_val), callbacks=[early_stopping])

model.fit_generator(train_generator, steps_per_epoch=e_step, epochs=50,
                    validation_data=val_generator,
                     validation_steps=v_step, shuffle=True, callbacks=[early_stopping])

# 모델 학습 과정 표시하기
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


# 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=t_step)
print("%s: %.2f%%" %(model.metics_names[1], scores[1]*100))

# 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=t_step)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)

from keras.models import load_model
model.saves('Flare_VGG16_power.h5')

