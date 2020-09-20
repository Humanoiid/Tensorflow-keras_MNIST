# Tensorflow-MNIST
 MNIST tensorflow for model comparision

# condition
1. python: 3.7.0
2. tensorflow: 2.1.0
3. keras: 2.3.1
4. GPU: Nvidia RTX 2060 Super
5. OS: Windows 10 2004

# Purpose and methods
1. As I want to compare the speed of prediction by trained models, i used MNIST dataset for each models
2. Every models use (224,224,3) color image. I resize original MNIST digitized numger data to (28x28 => 64x64) and stack 3 same data 3 for GRAY2RGB convert.
3. MNIST digit data and label was from tensorflow keras dataset.
4. Saved model was converted to tensorflow lite.
  * later it will move to this link: https://github.com/Humanoiid/Tensorflow_Lite_modelTest


# structure informations

## models

### from KERAS API
1. VGG19
2. VGG16
3. Resnet50
4. Resnet50V2
5. Resnet101
6. Resnet101V2
7. Resnet152
8. Resnet152V2
9. MobileNet
10. MobileNetV2
11. Alexnet

# Manual 
1. TF/MNIST_resize.ipynb : resize MNIST number data as we need. (here use 28x28 -> 64x64) and save at Desktop
  * need to configure resize parameter [int size]
  * change save directory on your computer environment. (if not change, it cannot save data or make odd folder)
2. by running test_note.ipnb, we can compare common image classification model on same condition
  * it is optimized to use same folder directory with previous part. (because of Load model and save result)
  * check parameters before run the model
    * model_name: can be working with given models above. (if the name is not in array, use AlexNet as default)
    * image size: same with 'size' parameter at 1.
    * channels: 1=gray, 3 = color. (this code only support 1 or 3. if not, there would be error)
    * batch: number of batch at learning
    * epoch: number of epoch at learning
  * for loop will automatically run function and print status.
    * learning progress is not visualized on notebook. but it can be checked in '~report_log.txt' file in save directory.
    * if not work, it shows error information in short. if you wan to see detail error information, take out function or progress and run again.
  * after finish learning well, there are four files for each models.
    * '~model_name~.h5': saved keras model file
    * '~model_name~_reportLog.txt': saved print from process. it shows model information and other informations
    * '~model_name~_Accuracy.png': saved keras model training history of accuracy
    * '~model_name~_Loss.png': saved keras model training history of Loss

# Issue
  * Accuracy seems low but, will be modified... someday...