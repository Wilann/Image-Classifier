# Deep Learning

## Project: Image Classifier 

### Source

Project 2 from Udacity's [Intro to Machine Learning Nanodegree](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)

### Description

Developed code for an image classifier built with PyTorch, then converted it into a command line application.

- Loaded training data, validation data, testing data, label mappings, and applied transformations (random scaling, cropping, resizing, flipping) to training data 
- Normalized means and standard deviations of all image color channels, shuffled data, and specified batch sizes
- Loaded pre-trained VGG16 network 
- Defined a new untrained feed-forward network as a classifier, using ReLU activations, and Dropout
- Defined Negative Log Likelihood Loss, Adam Optimizer, and learning rate 
- Trained the classifier layers with backpropagation in a CUDA GPU using pre-trained network to ~90% accuracy on validation set 
- Graphed training/validation/testing loss and validation/testing accuracy to ensure convergence to a global (or sufficient local) minimum
- Saved and loaded model to perform inference later 
- Preprocessed images (resize, crop, normalized means and standard deviations) to use as input for model testing 
- Visually displayed images to ensure preprocessing was successful 
- Predicted the class/label of an image using the trained model and plotted top 5 classes to ensure validity of prediction 

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org)
- [Torch & Torchvision](https://pytorch.org)

You will also need to have software installed to run and execute an [iPython Notebook](https://jupyter.org)

It's recommended to install [Anaconda](https://www.anaconda.com), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
