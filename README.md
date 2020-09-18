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
2. Every models use (224,224,3) color image. I multiplied to 3 for GRAY2RGB scaling and use upsampling layer (224/28 = 8) to save space for original data.
3. MNIST digit data and label was from tensorflow keras dataset.
4. Saved model was converted to tensorflow lite.
  * later it will move to this link: https://github.com/Humanoiid/Tensorflow_Lite_modelTest


# structure informations

## models
1. Resnet50
2. Resnet50v2
3. Resnet101
4. Resnet101v2
5. Resnet152
6. Resnet152v2

* v2 in 2,4,6 are based on keras API