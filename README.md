# GR5242 Final Project <br>

# Please access report through this link: https://docs.google.com/document/d/1vZJuP6cCw43J9asikQoB2q2AEsQDilBkyVxo2Uu2d14/edit?usp=sharing


## Team members:
- Yang Chen (yc3335@columbia.edu)
- 
- 
-

## Overview
The [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is an important image classification dataset. It consists of 60000 32x32 colour images in 10 classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks), with 6000 images per class. There are 50000 training images and 10000 test images.<br>

The **GOALS** of this project are to:
- Learn how to preprocess the image data by using simple baseline models
- Implement different architectures Convolutional Neural Networks (CNN) classifiers using GPU-enabled Tensorflow
- Here we pick VGG-14/16/19; Network in Network, Fractional pooling and Wide Residual Network 14X6/16X6
- Compare different CNN architectures pros and cons

**Tools:**
- GPU-enabled Tensorflow,colab

**Reproduce:**
- Two main folder for reproducing, **Baseline** and **Improve_Architecutres**, all of them could be reproduce on colab
- Data file is the CIFAR-10 by keras
- Output file contain the some of the captured images of tensorboard and the tensorboard event files
- The pdf file on the main pages is the **report** of our whole project, please grade with that **pdf** thanks!

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.


Reference: 
+ Model:https://github.com/BIGBALLON/cifar-10-cnn
+ Paper:http://www.columbia.edu/~kgh2122/papers/cifar.pdf
+ Last version:https://github.com/tianyiwangnova/Homework-at-Columbia-U
