# Flower Image Classifier And Python Command Line Tool

<div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
    <img src="/assets/Classification_Example.png" alt="Sample classification results for an image" width="500">
</div>

This project is split into two main parts:

- An image classifier for classifying various species of flowers based on images, using a neural network created in a Jupyter Notebook using PyTorch

- A command line application written in Python to use the image classifier in various instances

## Image Classifier

Typically, convolutional neural networks (CNN) are used for image classification problems. Among other reasons, they use filters to determine distinctive shapes in images which are especially helpful in classification.

In `Image Classifier Project.ipynb`, I used a pre-trained convolutional neural network (specifically VGG11) to obtain the convolutional filters normally used in CNNs to identify specific shapes in images. I then used a relatively smaller neural network as the classifier meant to determine the identity of each of a set of flowers based on previous images of different flower species.

During the training process, the extensive set of weights provided by VGG11 were frozen while the weights of the neural network classifier were updated and tuned so as to maintain a manageable training time. To mitigate overfitting of the neural network classifier to the training data set, I used `dropout` so that during various feedforward/backward passes of the training process, some nodes were frozen (not updated) with a certain probability at each pass. For internal layers, I used ReLU activation functions and for the final layer I used a log softmax function (as ReLU is not suitable for providing final classification probabilities).

After training the network and testing on a validation set and testing set, I saved the network as a checkpoint for later use without the need to re-train the model. I was able to achieve > 75% accuracy on the testing set using this architecture.

## Python Command Line Application

Using the Python application, a user may classify flower species by using the saved neural network or by setting various options and then training their own model.

Available options include:
* the number of best matching categories to return (e.g. for 3, the categories with the top 3 probabilities are shown)
* whether to run with GPU enabled (this will significantly speed up the training process)
* the number of training epochs
* the number of hidden units in the classifier
* the learning rate (i.e. step size for backpropagation)
* the type of architecture to be used for the pre-trained convolutional layers

## Notes

To run the Jupyter Notebook, you must have `Python` installed. You will also need to be able to access the Jupyter notebook; I recommend installing the Anaconda distribution, or for a smaller download, Miniconda. Both come with `Python` during installation. To run the notebook, use the command `jupyter-notebook` (depending on your version of `Python` and `Jupyter Notebook`, you may require some variation of this terminal command). This will open a browser window and the Jupyter Notebook dashboard, from which you can navigate to `Image Classifier Project.ipynb`.

This project was cloned from the starter code provided by Udacity's AI Programming with Python Nanodegree program.
