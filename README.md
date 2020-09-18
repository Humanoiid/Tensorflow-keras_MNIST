# Tensorflow-MNIST
 MNIST tensorflow for model comparision

# condition
1. python: 3.7.0
2. tensorflow: 2.3.0
3. GPU: Nvidia RTX 2060 Super
4. OS: Windows 10 2004

# Purpose and methods
1. As I want to compare the speed of prediction by trained models, i used MNIST dataset for each models
2. Every models use (224,224,3) color image. I multiplied to 3 for GRAY2RGB scaling and use upsampling layer (224/28 = 8) to save space for original data.
3. MNIST digit data and label was from tensorflow keras dataset.
4. Saved model was converted to tensorflow lite.
  * later it will move to this link: https://github.com/Humanoiid/Tensorflow_Lite_modelTest
