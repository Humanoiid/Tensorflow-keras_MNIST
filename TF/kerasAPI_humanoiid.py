
import argparse
import sys
import os
from datetime import datetime

# Tensorflow ans tf.keras
import tensorflow as tf
from tensorflow import keras
import tensorboard

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

# from tensorflow.keras.preprocessing import images

print(tf.__version__)

def func_keras_Models ( save_path, data_name, model_name, img_size, ch, bat_size, epoch_size, sys_stdout_backup ) :
    
    ### Paramters
    model_name_final = model_name + '_' + data_name + str(img_size) + '_b'+str(bat_size) +'_ep'+str(epoch_size)
    
    datasize_name = data_name
    datasize_name = data_name + '_' + str(img_size)
    data_path = save_path +datasize_name + '/'
    ext_npy = '.npy'
    
    ext_keras = '.h5'
    keras_model_path = save_path +'keras/' + model_name_final + ext_keras 
    
    
    
    # tensorboard setting
    # link: https://rfriend.tistory.com/554
    # https://m.blog.naver.com/unisun2/221679506723
#     logdir=save_path +'keras/' + ".logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = (save_path +'keras/' + ".logs/" + model_name_final +'/'+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    tf.io.gfile.makedirs(logdir)
    model_callback = [tf.keras.callbacks.EarlyStopping(patience=2),
                      tf.keras.callbacks.ModelCheckpoint(filepath=save_path +'keras/' + ".logs/" + model_name_final +'/'+'model.{epoch:02d}-{val_loss:.2f}.h5'),
                      tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, profile_batch=0, update_freq=1000)
                     ]
    
    

    
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

    
#     plt.figure(figsize=(10,10))
#     for i in range(25):
#         plt.subplot(5,5,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(X_train[i], cmap=plt.cm.binary)
#         plt.xlabel(y_train[i])
#         plt.colorbar()    
#     plt.show()
    
    
    tf.keras.backend.clear_session()
    ################### System
    print('[Load data. Done.] ', file=sys_stdout_backup)
    
    ### Model
    def_input = Input(shape=(img_size,img_size,3))
    if(model_name == 'VGG19'):
        # https://arxiv.org/pdf/1409.1556.pdf
        try:
            model =  tf.keras.applications.VGG19(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return
        
    elif(model_name == 'VGG16'):
        try:
            model =  tf.keras.applications.VGG16(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet50'):
        # https://arxiv.org/pdf/1512.03385.pdf
        try:
            model =  tf.keras.applications.ResNet50(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet50V2'):
        # https://arxiv.org/pdf/1603.05027.pdf
        try:
            model =  tf.keras.applications.ResNet50V2(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet101'):
        try:
            model =  tf.keras.applications.ResNet101(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet101V2'):
        try:
            model =  tf.keras.applications.ResNet101V2(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet152'):
        try:
            model =  tf.keras.applications.ResNet152(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'ResNet152V2'):
        try:
            model =  tf.keras.applications.ResNet152V2(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'MobileNet'):
        # https://arxiv.org/pdf/1704.04861.pdf
        try:
            model =  tf.keras.applications.MobileNet(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'MobileNetV2'):
        # https://arxiv.org/pdf/1801.04381.pdf
        try:
            model =  tf.keras.applications.MobileNetV2(input_tensor=def_input, weights=None, classes=n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return            
            
    elif(model_name == 'AlexNet'):
        try:
            model = AlexNet(img_size, n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return
    else:
        try:
            model = AlexNet(img_size, n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            return
#         print("test model(just simple fast model)")
#         model = Sequential() # 순차 모델 생성
#         model.add(Conv2D(32, # 필터 수
#           kernel_size=(3, 3), # 필터 사이즈
#           activation='relu', # 활성화 함수
#           input_tensor=def_input)) # (입력 이미지 사이즈, 채널 수)
#         model.add(Conv2D(64,kernel_size=(3, 3), activation='relu')) # 은닉층
#         model.add(MaxPooling2D(pool_size=(2, 2))) # 사소한 변화 무시
#         model.add(Flatten()) # 영상 -> 문자열 변환
#         model.add(Dense(128, activation='relu')) # 은닉층
#         model.add(Dense(5, activation='softmax')) # 최종 출력층

#         model.add(layers.Dense(32, activation='relu', input_tensor = def_input))

#         model.add(layers.Dense(Conv2D(32, # 필터 수
#           kernel_size=(8, 8), # 필터 사이즈
#           activation='relu', # 활성화 함수
# #           input_tensor=def_input))) # (입력 이미지 사이즈, 채널 수)
#           input_size=(img_size,img_size,3)))) # (입력 이미지 사이즈, 채널 수)
#         model.add(layers.Dense(n_classes, activation='softmax'))
        
    ## model summary save
    print('[model summary]==============')
    print(model.summary())
    
    
    ################### System
    print('[Load Model. Done.] ',file=sys_stdout_backup)

    # 3. 모델 학습과정 설정하기
    # model.compile(loss='categorical_crossentropy', optimizer='adam',
    #               metrics=['accuracy']) # 최적화 알고리즘 설정
#     opt = 'adam'
    opt = 'sgd'
    los = 'categorical_crossentropy'
    met = ['accuracy']
    model.compile(optimizer=opt,
                    loss=los,
                    metrics=met)
    print("Optimizer: "+ opt)
    print("loss: "+ los)
    print("metrics: "+str (met))
    
    
    ################### System
    print('[Model Complie. Done.] ',file=sys_stdout_backup)
    
    

    
    
    # 4. 모델 학습시키기
    # model.fit_generator(
    #  train_generator, # 훈련셋 지정
    #  steps_per_epoch=200, # 총 훈련셋 수 / 배치 사이즈 (= 1000/50)
    #  epochs=150) # 전체 훈련셋 학습 반복 횟수 지정
    training_history = model.fit(X_train, y_train, validation_split=0.02, epochs=epoch_size, batch_size=bat_size, verbose=0, callbacks=model_callback)
#     print('## training loss and acc ##')
#     print(hist.history['loss'])
#     print(hist.history['accuracy'])

    ################### System
    print('[Training. Done.] ', file=sys_stdout_backup)
    
    # save model 
    model.save(keras_model_path)
    finish_training = datetime.now()
    print("finish training date and time : " + str(finish_training))    


    # model evaluation
    # link: https://tykimos.github.io/2017/06/10/Model_Save_Load/
    result = model.evaluate(X_test, y_test, batch_size=bat_size)
    print('')
    print('loss_and_metrics : ' + str(result))
    print('', file=sys_stdout_backup)
    print('loss_and_metrics : ' + str(result), file=sys_stdout_backup)

    finish_evaluation = datetime.now()
    print("finish evaluation date and time : " + str(finish_evaluation))

    # save history
    plt.figure(1)
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'valid'], loc='upper left')
    plt.savefig(save_path +'keras/' + model_name_final + '_Accuracy.png', dpi=300)
    plt.show()
    
    plt.figure(2)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'valid'], loc='upper left')
    plt.savefig(save_path +'keras/' + model_name_final + '_Loss.png', dpi=300)
    plt.show()
    
    
    save_history = datetime.now()
    print("save history date and time : " + str(save_history))
    
    print('====Time information ===')
    print("Start date and time : " + str(start))
    print("finish training date and time : " + str(finish_training))  
    print("finish evaluation date and time : " + str(finish_evaluation))
    print("save history date and time : " + str(save_history))

    
    

    
    print('====Time information ===')
    print("model Start date and time : " + str(start), file=sys_stdout_backup)
    print("finish training date and time : " + str(finish_training), file=sys_stdout_backup)  
    print("finish evaluation date and time : " + str(finish_evaluation), file=sys_stdout_backup)
    print("save history date and time : " + str(save_history), file=sys_stdout_backup)
    
    return
    
    
    
def AlexNet(image_size, number_classes):
    # https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
    # https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    model = models.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), 
                            activation='relu', input_shape=(image_size,image_size,3)))
    
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    
    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    
    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(number_classes, activation='softmax'))

    return model

