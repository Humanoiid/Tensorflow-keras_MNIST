# resnet 101v2

import argparse
import sys

# Tensorflow ans tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.transform import resize

# keras libraries
from tensorflow.keras import models
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, UpSampling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


# resnet50
# from keras.applications import ResNet50
# from keras.applications.resnet50 import ResNet50




print(tf.__version__)


# # https://www.tensorflow.org/guide/gpu?hl=ko#gpu_%EB%A9%94%EB%AA%A8%EB%A6%AC_%EC%A0%9C%ED%95%9C%ED%95%98%EA%B8%B0
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
#   except RuntimeError as e:
#     # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#     print(e)



model_name = 'mnist_resnet101v2_model' # should update this value by each model

keras_name = 'save/keras/'
# keras2tf_name = 'save/keras-tf/'
# tf2tfl_name = 'save/tf-tfl/'
tfl_name = 'save/tflite/'

ext_keras = '.h5'
ext_tfl = '.tflite'

temp = keras_name+model_name + ext_keras 
print(temp)



    
    
(X_train_org, y_train_org), (X_test_org, y_test_org) = tf.keras.datasets.mnist.load_data()
print(y_train_org[0])


# X_train_224 = resize(X_train_org, (begin():end(),224,224))
print(type(X_train_org.shape))
num_train,x_org, y_org = X_train_org.shape
num_test,x_org, y_org = X_test_org.shape

X_train = X_train_org.astype(np.float32) / 255.0
X_test = X_test_org.astype(np.float32) / 255.0

# number of classes (label. 0~9: 10)
n_classes = 10

y_train = tf.keras.utils.to_categorical(y_train_org, n_classes) ### label (class vector(integer) => binary class matrix): 범주형 인코딩
y_test = tf.keras.utils.to_categorical(y_test_org, n_classes) 
print(y_train[0])
y_train = y_train.astype(np.int32) # label => 
y_test = y_test.astype(np.int32)
print(y_train[0])

X_train_rgb = np.stack((X_train,)*3, axis=-1)
X_test_rgb = np.stack((X_test,)*3, axis=-1)
print(type(X_train))
print(X_train.shape)
print("===")
print(type(X_train_rgb))
print(X_train_rgb.shape)
print("===")
print(y_train_org.shape)

# model
# def_input = Input(shape=(28,28,1))
def_input = Input(shape=(28,28,3))
ratio = int(224/28)
upsampled = UpSampling2D((ratio, ratio))(def_input)
# model =  tf.keras.applications.ResNet101(input_tensor=upsampled, weights=None, classes=n_classes)
model =  tf.keras.applications.resnet.ResNet101v2(input_tensor=upsampled, weights=None, classes=n_classes)
# model.add(Dense(n_classes, activation='softmax'))

print("=================")
print("=================")
print(model.summary())
print("=================")
print("=================")

# 3. 모델 학습과정 설정하기
# model.compile(loss='categorical_crossentropy', optimizer='adam',
#               metrics=['accuracy']) # 최적화 알고리즘 설정

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
                
# 4. 모델 학습시키기
# model.fit_generator(
#  train_generator, # 훈련셋 지정
#  steps_per_epoch=200, # 총 훈련셋 수 / 배치 사이즈 (= 1000/50)
#  epochs=150) # 전체 훈련셋 학습 반복 횟수 지정
model.fit(X_train_rgb, y_train, epochs=5, batch_size=32)

# model evaluation
# link: https://tykimos.github.io/2017/06/10/Model_Save_Load/
loss_and_metrics = model.evaluate(X_test_rgb, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))


# save model
model.save(temp)