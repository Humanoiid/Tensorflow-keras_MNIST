import argparse
import sys
import os
from datetime import datetime
import pyautogui


# Tensorflow ans tf.keras
import tensorflow as tf
import tensorboard
from tensorflow import keras
from keras import backend as K

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


# from keras import backend as K
# K.set_image_dim_ordering('tf')
# from tensorflow.keras.preprocessing import images

print(tf.__version__)

def func_keras_Models ( save_path, data_name, model_name, img_size, ch, bat_size, epoch_size, sys_stdout_backup, reportLog,
                       num_filter, num_blocks, num_fullyCon, n_classes, ab_type) :
    
    ### Paramters
    model_name_final = model_name + '_' + data_name + str(img_size) + '_b'+str(bat_size) +'_ep' + str(epoch_size) + '_fi' + str(num_filter) + '_bl' + str(num_blocks) + '_fc' + str(num_fullyCon)
    
    datasize_name = data_name
    datasize_name = data_name + '_' + str(img_size)
    data_path = save_path +datasize_name + '/'
    ext_npy = '.npy'
    
    ext_keras = '.h5'
    keras_model_path = save_path +'keras/' + model_name + ab_type + '/' + model_name_final + ext_keras 
    
    
    
    # tensorboard setting
    # link: https://rfriend.tistory.com/554
    # https://m.blog.naver.com/unisun2/221679506723
#     logdir=save_path +'keras/' + ".logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = (save_path +'keras/'+ model_name + ab_type + '/' + ".logs/" + model_name_final +'/'+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    tf.io.gfile.makedirs(logdir)
#     model_callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=2),
# #                       tf.keras.callbacks.ModelCheckpoint(filepath=save_path +'keras/'+ model_name + ab_type + '/' + ".logs/" + model_name_final +'/'+'model.{epoch:02d}-{val_loss:.2f}.h5'),
#                       tf.keras.callbacks.ModelCheckpoint(filepath=save_path +'keras/'+ model_name + ab_type + '/' + ".logs/" + model_name_final +'/'+'best_model.h5', monitor='val_loss', mode='min', save_best_only=True),
#                       tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, profile_batch=0, update_freq=1000)
#                      ]
    log_path = save_path +'keras/'+ model_name + ab_type + '/' + ".logs/" + model_name_final +'/'
#     model_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=log_path+'model.best_model.h5',monitor='loss', verbose=1, save_best_only=True,save_freq='epoch'),
#                       tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3),
#                       tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, profile_batch=0, update_freq=1000)
#                      ]
    model_callback = [tf.keras.callbacks.EarlyStopping(patience=2),
                      tf.keras.callbacks.ModelCheckpoint(filepath=log_path+'model.{epoch:02d}-{val_loss:.2f}.h5'),
                      tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_images=True, profile_batch=0, update_freq=1000)
                     ]
#                       tf.keras.callbacks.ModelCheckpoint(filepath=log_path+'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True),
#     tf.keras.callbacks.ModelCheckpoint(filepath=log_path+'model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', verbose=1, save_best_only=True),
    

    
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

    
    
    tf.keras.backend.clear_session()
    ################### System
    print('[Load data. Done.] ', file=sys_stdout_backup)
    
    ### Model
    def_input = Input(shape=(img_size,img_size,3))
    if(model_name == 'VGG19'):
        # https://arxiv.org/pdf/1409.1556.pdf
        try:
            model = VGG19(img_size, num_filter, num_blocks, num_fullyCon, n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            tf.keras.backend.clear_session()
            return
        
    elif(model_name == 'ResNet152'):
        try:
            model = ResNet152(def_input, num_filter, num_blocks, num_fullyCon, n_classes)
        except Exception as ex:
            print(type(ex))
            print('Error information: ',ex)
            print(type(ex), file=sys_stdout_backup)
            print('Error information: ',ex, file=sys_stdout_backup)
            print('Model Error. skip', file=sys_stdout_backup)
            tf.keras.backend.clear_session()
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
            tf.keras.backend.clear_session()
            return
                    
    else:
        
        print('\n No specific model. return the process' ,file=sys_stdout_backup)
        return
        
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
#     met = [keras.metrics.Accuracy()]
    #'loss','accuracy','val_loss','val_accuracy']
    model.compile(optimizer=opt,
                    loss=los,
                    metrics=met)
    print("Optimizer: "+ opt)
    print("loss: "+ los)
    print("metrics: "+str (met))
    
    
    ################### System
    print('[Model Complie. Done.] ',file=sys_stdout_backup)
    
    

    
    
#     # system out 바꾸기.
#     sys_stdout_backup
#     reportLog
    
    sys.stdout = sys_stdout_backup
    
#     4. 모델 학습시키기
#     model.fit_generator(
#      train_generator, # 훈련셋 지정
#      steps_per_epoch=200, # 총 훈련셋 수 / 배치 사이즈 (= 1000/50)
#      epochs=150) # 전체 훈련셋 학습 반복 횟수 지정
#     training_history = model.fit(X_train, y_train, validation_split=0.02, epochs=epoch_size, batch_size=bat_size, verbose=1, callbacks=model_callback)

#     training_history = model.fit(X_train, validation_data=(X_test, y_test), epochs=epoch_size, batch_size=bat_size, verbose=1, callbacks=model_callback)

    # TEST
    training_history = model.fit(X_train[1:20], y_train[1:20], validation_split=0.1, epochs=epoch_size, batch_size=bat_size, verbose=1, callbacks=model_callback)

#     print('## training loss and acc ##')
#     print(hist.history['loss'])
#     print(hist.history['accuracy'])


    sys.stdout = reportLog
    
    
    ################### System
    print('[Training. Done.] ', file=sys_stdout_backup)
    
    # save model 
    model.save(keras_model_path)
    finish_training = datetime.now()
    print("finish training date and time : " + str(finish_training))    


    # model evaluation
    # link: https://tykimos.github.io/2017/06/10/Model_Save_Load/
    result = model.evaluate(X_test, y_test, batch_size=bat_size, verbose=2)
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
    plt.savefig(save_path +'keras/'+ model_name + ab_type + '/' + model_name_final + '_Accuracy.png', dpi=300)
    plt.show()
    
    plt.figure(2)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'valid'], loc='upper left')
    plt.savefig(save_path +'keras/'+ model_name + ab_type + '/' + model_name_final + '_Loss.png', dpi=300)
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
    tf.keras.backend.clear_session()
    return
    
    
    
def VGG19(img_size = 64, num_filter = 1, num_blocks = 5, num_fullyCon = 2, number_classes = 10):
   # VGG 19
    tf.keras.backend.clear_session()

    model = models.Sequential()

    start_filter = 64
    applied_filter=start_filter*num_filter
    # Block 1
    if (num_blocks >= 1):
        model.add(layers.Conv2D(filters=int(applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", 
                                input_shape=(img_size,img_size,3)))
        model.add(layers.Conv2D(filters=int(applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 2
    if (num_blocks >= 2):
        model.add(layers.Conv2D(filters=int(2*applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.Conv2D(filters=int(2*applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block 3
    if (num_blocks >= 3):
        for j in range(4):
            model.add(layers.Conv2D(filters=int((2**2)*applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Block Loop
    for i in range (3,num_blocks): # 64ch * 8 fixed on paper.
        for j in range(4):
            model.add(layers.Conv2D(filters=int((2**3)*applied_filter), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(layers.Flatten())
    for i in range(num_fullyCon): 
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))

    model.add(layers.Dense(number_classes, activation='softmax'))
    return model

def resnetBlock_seq (model, filt, loop, stride=2):
    for i in range(1, loop + 1):
        if i == 1:
            model.add(layers.Conv2D(filters=int(filt), kernel_size=(1,1), strides=(stride,stride), activation='relu', padding="same"))
        else:
            model.add(layers.Conv2D(filters=int(filt), kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=int(filt), kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters=int(4*filt), kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"))    
        model.add(layers.BatchNormalization())

    return model

# ResNet152_seq
def ResNet152_seq(img_size = 64, num_filter = 1, num_blocks = 5, num_fullyCon = 0, number_classes = 10):
    model = models.Sequential()

    start_filter = 64
    applied_filter=start_filter*num_filter
    ## Common PArt
    # Conv 1
    if (num_blocks >= 1):
        model.add(layers.Conv2D(filters=int(applied_filter), kernel_size=(7,7), strides=(2,2), activation='relu', padding="same", 
                                input_shape=(img_size,img_size,3)))
        model.add(layers.BatchNormalization())
    # Conv 2
    if (num_blocks >= 2):
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    if (num_blocks >= 2):
        resnetBlock_seq(model, applied_filter, 3, 1)

    # Conv 3
    if (num_blocks >= 3):
        resnetBlock_seq(model, 2*applied_filter, 8, 2)
    # Conv 4
    if (num_blocks >= 4):
        resnetBlock_seq(model, 4*applied_filter, 36, 2)
    # Conv 5
    if (num_blocks >= 5):
        resnetBlock_seq(model, 8*applied_filter, 3, 2)    


    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Flatten())
    i = 0
    while(i != num_fullyCon):
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))


    return model
    
def resnetBlock_func (x, filt, loop, stride=2):
    for i in range(1, loop + 1):
        
#         shortcut = x
        if i == 1:
            x = layers.Conv2D(int(filt), kernel_size=(1, 1), strides=(stride, stride), padding="same")(x)
            shortcut = layers.Conv2D(int(4*filt), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
            shortcut = BatchNormalization()(shortcut) 
        else:
            x = layers.Conv2D(int(filt), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        
        x = layers.Conv2D(int(filt), kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(int(4*filt), kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)

        
        x = layers.Add()([x, shortcut])    
        x = layers.Activation('relu')(x)
 
        shortcut = x   

    return x

def ResNet152(def_input, num_filter = 1, num_blocks = 5, num_fullyCon = 0, number_classes = 10):
# ResNet152
    start_filter = 64
    applied_filter=start_filter*num_filter

    #functional

    ## Common PArt
    # Conv 1
    if (num_blocks >= 1):
        x = layers.Conv2D(int(applied_filter), kernel_size=(7, 7), strides=(2, 2), padding="same")(def_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Conv 2
    if (num_blocks >= 2):
        x = layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)  
    if (num_blocks >= 2):
        x=resnetBlock_func(x, applied_filter, 3, 1)


    # Conv 3
    if (num_blocks >= 3):
        x=resnetBlock_func(x, 2*applied_filter, 8, 2)
    # Conv 4
    if (num_blocks >= 4):
        x=resnetBlock_func(x, 4*applied_filter, 36, 2)
    # Conv 5
    if (num_blocks >= 5):
        x=resnetBlock_func(x, 8*applied_filter, 3, 2)    



    x = layers.GlobalAveragePooling2D()(x)
    output_tensor = layers.Dense(number_classes,activation='softmax')(x)
    model = Model(def_input, output_tensor)


    return model