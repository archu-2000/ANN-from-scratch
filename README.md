# ANN-from-scratch
Designing Artificial Neural Networks for classification of LBW Cases from Scratch using Numpy

Pre-processing
     The dataset is read by using pandas library with the help of dataframe. The missing values of the dataset is filled with their means across the columns. The values in the dataset are standardized to have a mean of zero and a standard deviation of one (unit variance).
We have used Scikit library for generating test-train splits

Train-test splitting 
     Dataset is split into training (70 %) and testing (30 %) sets. Only the training set will be used for tuning the neural networks and testing set will be used for performance evaluation.

Hyperparameters used:
- learning_rate : To train the neural network 
- layers : Performs the nonlinear transformations of the inputs.
- neural_net: list representation of the fully connected neural network  
- activation(Activation function) : A function that is added into an artificial neural network in order to help the network learn complex patterns in the data.Here ,we have used sigmoid and ReLU functions.
- weights (W): weights associated with each layer.
- biases(b): the biases associated with each layer.

Detailed steps to run the File :
	The command to run the file is python3 ANN_model.py
        The dataset is first preprocessed using the preprocessing code. The preprocessed dataset is used in the neural network code. The files are read using pd.read_csv() and the dataset files NEED NOT be passed as an argument while running the code. 

Key features:
        We have totally 5 layers (input layer, 3 hidden layers and output layer) in our neural network model which gives an accuracy of about 98% for the training dataset and around 86% on the testing dataset. We have represented each layer as an object of a class which makes it easier to access the features associated with each layer. Our model also allows using three different activation functions.
