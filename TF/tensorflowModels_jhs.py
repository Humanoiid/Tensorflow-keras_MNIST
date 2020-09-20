
import argparse
import sys
import os
from datetime import datetime

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

print(tf.__version__)

def func_tf_models ( save_path, data_name, model_name, img_size, ch, bat_size, epoch_size ) :
    
    ### Paramters
    model_name_final = model_name + '_' + data_name + str(img_size) + '_b'+str(bat_size) +'_ep'+str(epoch_size)
    
    datasize_name = data_name
    datasize_name = data_name + '_' + str(img_size)
    data_path = save_path +datasize_name + '/'
    ext_npy = '.npy'
    
    ext_keras = '.h5'
    ext_tf = '.tf'
    keras_model_path = save_path +'keras/' + model_name_final + ext_keras 
    
    
    ### save Path
    if(os.path.isdir(save_path) != True):
        os.mkdir(save_path)
    if(os.path.isdir(save_path + '/keras') != True):
        os.mkdir(save_path + '/keras')  
    # progress report
    ## model summary save
    history = open(save_path + 'keras/' + model_name_final + '_history.txt', 'w')
    sys_stdout_backup = sys.stdout
    sys.stdout = history
    
    start = datetime.now()
    print("Start date and time : " + str(start))
    
    # number of classes (label. 0~9: 10)
    if(data_name == 'MNIST'):
        n_classes = 10
    
    
    ### Load data
    X_train = np.load(data_path + datasize_name + 'Xtrain' + ext_npy)
    X_test = np.load(data_path + datasize_name + 'Xtest' + ext_npy)
    y_train = np.load(data_path + datasize_name + 'ytrain' + ext_npy)
    y_test = np.load(data_path + datasize_name + 'ytest' + ext_npy)


    
    ### Preprocessing
    num_train,num_x, num_y = X_train.shape
    num_test,num_x, num_y = X_test.shape

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    
    y_train = tf.keras.utils.to_categorical(y_train, n_classes) ### label (class vector(integer) => binary class matrix): 범주형 인코딩
    y_test = tf.keras.utils.to_categorical(y_test, n_classes) 
    y_train = y_train.astype(np.int32) # label => 
    y_test = y_test.astype(np.int32)

    print("[Data information]==============")
    print("================================")
    print('X_train information: '+ str(X_train.shape))
    print('X_test information: '+ str(X_test.shape))
    print("===")
    if (ch == 1):
        X_train = np.stack((X_train,)*3, axis=-1)
        X_test = np.stack((X_test,)*3, axis=-1)
        print('X_train gray2rgb information: '+str(X_train.shape))
        print('X_test gray2rgb information: '+str(X_test.shape))
    print("================================")


    ### Model
    def_input = Input(shape=(img_size,img_size,3))
    
    if(model_name == 'VGG19'):
#         model =  tf.keras.applications.VGG19(input_tensor=def_input, weights=None, pooling='max', classes=n_classes, classifier_activation="softmax")
        model =  tf.keras.applications.VGG19(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'VGG16'):
        model =  tf.keras.applications.VGG16(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet50'):
        model =  tf.keras.applications.ResNet50(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet50V2'):
        model =  tf.keras.applications.ResNet50V2(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet101'):
        model =  tf.keras.applications.ResNet101(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet101V2'):
        model =  tf.keras.applications.ResNet101V2(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet152'):
        model =  tf.keras.applications.ResNet152(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'ResNet152V2'):
        model =  tf.keras.applications.ResNet152V2(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'MobileNet'):
        model =  tf.keras.applications.MobileNet(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    elif(model_name == 'MobileNetV2'):
        model =  tf.keras.applications.MobileNetV2(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax")
    else:
        print("not prepared model. use mobilenetV2 as default model")
        model =  tf.keras.applications.MobileNetV2(input_tensor=def_input, weights=None, classes=n_classes)
#         model = Sequential() # 순차 모델 생성
#         model.add(Conv2D(32, # 필터 수
#           kernel_size=(3, 3), # 필터 사이즈
#           activation='relu', # 활성화 함수
#           input_tensor=def_input) # (입력 이미지 사이즈, 채널 수)
#         model.add(Conv2D(64,kernel_size=(3, 3), activation='relu')) # 은닉층
#         model.add(MaxPooling2D(pool_size=(2, 2))) # 사소한 변화 무시
#         model.add(Flatten()) # 영상 -> 문자열 변환
#         model.add(Dense(128, activation='relu')) # 은닉층
#         model.add(Dense(5, activation='softmax')) # 최종 출력층
        
    ## model summary save
    print("[model summary]==============")
    print(model.summary())
    

    # 3. 모델 학습과정 설정하기
    # model.compile(loss='categorical_crossentropy', optimizer='adam',
    #               metrics=['accuracy']) # 최적화 알고리즘 설정
    opt = 'adam'
    los = 'categorical_crossentropy'
    met = ['accuracy']
    model.compile(optimizer=opt,
                    loss=los,
                    metrics=met)
    print("Optimizer: "+ opt)
    print("loss: "+ los)
    print("metrics: "+str (met))
    # 4. 모델 학습시키기
    # model.fit_generator(
    #  train_generator, # 훈련셋 지정
    #  steps_per_epoch=200, # 총 훈련셋 수 / 배치 사이즈 (= 1000/50)
    #  epochs=150) # 전체 훈련셋 학습 반복 횟수 지정
    model.fit(X_train, y_train, epochs=epoch_size, batch_size=bat_size)

    # model evaluation
    # link: https://tykimos.github.io/2017/06/10/Model_Save_Load/
    result = model.evaluate(X_test, y_test, batch_size=bat_size)
    print('')
    print('loss_and_metrics : ' + str(result))


    # save model 
    model.save(keras_model_path)
    finish = datetime.now()
    print("finish date and time : " + str(finish))
    
    
    
    


    
    sys.stdout = sys_stdout_backup
    history.close()
    
def vgg19(input_tensor=def_input, weights=None, classes=n_classes, classifier_activation="softmax"):
    