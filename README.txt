Designing Artificial Neural Networks for classification of LBW Cases from Scratch 




Implementation

The neural network is trained for some number of epochs and few hyperparameters.


Pre-processing


The dataset is read by using pandas library with the help of dataframe. The missing values of the dataset is filled with their means across the columns. The values in the dataset are standardized to have a mean of zero and a standard deviation of one (unit variance).
We have used Scikit library for generating test-train splits




Train-test splitting 


Dataset is split into training (70 %) and testing (30 %) sets. Only the training set will be used for tuning the neural networks and testing set will be used for performance evaluation.
We substantiate a layer using class in python. Every layer (except the input layer) has a weight matrix W, a bias vector b, and an activation function. Each layer is appended to a list called neural_net. That list would then be a representation of your fully connected neural network.
         We do a sanity check on the number of hyperparameters. The number of datums available should exceed the number of hyperparameters, otherwise it will overfit. 
For forward propagation, we have defined a predict() function given a certain set of weights and biases.
We have used sigmoid and ReLU for activation functions. The ReLU function is used for the hidden layers and sigmoid function for the output layer.Activation functions are defined one by one. ReLU (Rectified linear unit), the main idea behind the ReLU is to perform a threshold operation to each input element where values less than zero are set to zero. Sigmoid function takes in real numbers in any range and returns a real-valued output.
We have defined the gradient descent function that basically performs as a building block for training a neural network. Here we try to minimize the loss metric iteratively by gradient descent function. 
In the backward propagation once we have defined a loss metric for evaluating performance, this acknowledges how  loss metric changes ,when we  unsettle each weight and bias.
Finally the confusion matrix will aid us in being able to visually observe how well a neural network is predicting during inference.So confusion matrix is filled up based on the test set whose true labels is known .When the test data is passed its predictions are noted.A table of predicted labels vs true labels is then filled out.






Hyperparameters used:


learning_rate : To train the neural network 
layers : Performs the nonlinear transformations of the inputs.
neural_net: list representation of the fully connected neural network  
activation(Activation function) : A function that is added into an artificial neural network in order to help the network learn complex patterns in the data.Here ,we have used sigmoid and ReLU functions.
weights (W): weights associated with each layer.
biases(b): the biases associated with each layer.


Detailed steps to run the File :

	The command to run the file is python3 ANN_model.py
        The dataset is first preprocessed using the preprocessing code. The preprocessed dataset is used in the neural network code. The files are read using pd.read_csv() and the dataset files NEED NOT be passed as an argument while running the code. 




Key feature of our design that makes it stand out:
        We have totally 5 layers (input layer, 3 hidden layers and output layer) in our neural network model which gives an accuracy of about 98% for the training dataset and around 86% on the testing dataset. We have represented each layer as an object of a class which makes it easier to access the features associated with each layer. Our model also allows using three different activation functions.