You've heard the term overfitting a number of times to this point. Overfitting is simply the concept of being over specialized in training -- namely that your model is very good at classifying what it is trained for, but not so good at classifying things that it hasn't seen. 
In order to generalize your model more effectively, you will of course need a greater breadth of samples to train it on. That's not always possible, but a nice potential shortcut to this is Image Augmentation, where you tweak the training set to potentially increase the diversity of subjects it covers.

Learning Objectives:
- Recognize the impact of adding image augmentation to the training process, particularly in time
- Demonstrate overfitting or lack of by plotting training and validation accuracies
- Familiarize with the ImageDataGenerator parameters used for carrying out image augmentation
- Learn how to mitigate overfitting by using data augmentation techniques.
