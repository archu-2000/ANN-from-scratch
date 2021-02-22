'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

#import all the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset():
    df1=pd.read_csv(r'LBW_preprocessed.csv')        #reads the preprocessed dataset
    x=df1.iloc[:,:-1]           #the input features/columns 
    y=df1.iloc[:,-1]            #the target column
    x=np.array(x)
    y=np.array(y)
    y=y.reshape(y.shape[0],1)   
    return (x,y)

x,y=read_dataset()        #calls the read_dataset function

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=1)      #splits the dataset into training and testing dataset

#reshape the training and testing dataset to their transpose

#print(xTrain.shape)            #(67, 9)
xTrain = xTrain.reshape(9, 67)          
#print(yTrain.shape)            #(67, 1)
yTrain = yTrain.reshape(1, 67)
#print(xTest.shape)             #(29,9)
xTest = xTest.reshape(9, 29)
#print(yTest.shape)             #(29,1)
yTest = yTest.reshape(1, 29)

#print the shapes of the training and testing dataset
print('xtrain: ',xTrain.shape)
print('ytrain: ',yTrain.shape)
print('xtest: ',xTest.shape)
print('ytest: ',yTest.shape)

#this function is used for sigmoid activation function. 
def sigmoid_act(inp):
    s=1/(1+np.exp(-1*inp))
    return s

#this function is used for sigmoid deactivation function.
def sigmoid_deact(inp):
    s=sigmoid(inp)*(1-sigmoid(inp))
    return s

#a class to represent each layer.
class layer:
    def __init__(self, layer_index, input_dim, output_dim, activation):
        self.layer_index=layer_index            #tells which layer it is
        self.input_dim=input_dim                #represents the dimension of the input to this layer. For the input layer the input_dim is 0 and for the other layers it is the output_dim of the previous layer.
        self.output_dim=output_dim              #represents the dimension of the output of this layer.
        self.activation=activation              #represents the activation function of the respective layer.
        if layer_index!=0:                      #not an input layer
            np.random.seed(0)                       
            self.W=np.random.randn(output_dim,input_dim)*np.sqrt(2/input_dim)           #initialize the weights for the corresponding layer randomly
            self.b=np.random.randn(output_dim,1)*np.sqrt(2/input_dim)                   #initialize the biases for the corresponding layer randomly


#a class to represent the neural network model
class NN:
    #initialize the parameters for the model
    def __init__(self):
        self.layers=[]          #represents the list of layers of this model
        self.x_rows=0           #represents the number of rows in the input features dataset
        self.x_cols=0           #represents the number of columns in the input features dataset
        self.y_rows=0           #represents the number of rows in the output features dataset
        self.y_cols=0           #represents the number of columns in the output features dataset
        self.neural_net=[]      #list representation of the fully connected neural network
        self.learning_rate=0    #initialize the learning rate for the model
        
    #Function that initializes all the layers and the neural_net list
    def fit(self,X,Y):                                  
        self.x_rows, self.x_cols=X.shape
        self.y_rows, self.y_cols=Y.shape
        self.layers=[self.x_rows, 9, 8, 7, self.y_rows]         #assign the layers of this model where each value in this list represents the number of neurons in each layer
        for i in range(len(self.layers)):
            if i==0:                            #input layer
                self.neural_net.append(layer(i, 0, self.layers[i],activation=''))
            elif i+1==len(self.layers):         #output layer
                self.neural_net.append(layer(i, self.layers[i-1],self.layers[i],activation='sigmoid'))     #sigmoid activation function for the output layer
            else:                               #hidden layers
                self.neural_net.append(layer(i, self.layers[i-1],self.layers[i],activation='relu'))        #relu activation function for the input layer
        
	
    #Function that checks if it will overfit
    def check_overfit(self):
        pred_n_param=sum([(self.layers[i]+1)*self.layers[i+1] for i in range(len(self.layers)-1)])
        act_n_param = sum([self.neural_net[i].W.size+self.neural_net[i].b.size for i in range(1,len(self.layers))])
        data=self.x_rows*self.x_cols
        print('Predicted number of hyperparameters: ',pred_n_param)
        print('Actual number of hyperparameters: ',act_n_param)
        print('Number of data: ',data)
        if act_n_param>=data:
            return -1
        return 1

    #The activation function that takes in the output from the summation function of the corresponding layer and the activation function assigned to that layer
    def activation(self, inp, act_func):
        #relu function
        if act_func=='relu':
            return np.maximum(inp, np.zeros(inp.shape))
        #sigmoid function
        elif act_func=='sigmoid':
            return sigmoid_act(inp)
        #linear function
        elif act_func=='linear':
            return inp

    #The deactivation function used for backward propagation
    def deactivation(self, inp, act_func):
        #relu function
        if act_func=='relu':
            return np.maximum(np.sign(inp), np.zeros(inp.shape))
        #sigmoid function
        elif act_func=='sigmoid':
            return sigmoid_deact(inp)
        #linear function
        elif act_func=='linear':
            return np.ones(inp.shape)

    #The predict function performs a simple feed forward of weights and outputs yhat values. yhat is a list of the predicted value for df X
    def predict(self,X):
        self.neural_net[0].A=X
        for i in range(1,len(self.layers)):
            self.neural_net[i].Z=np.add(np.dot(self.neural_net[i].W,self.neural_net[i-1].A), self.neural_net[i].b)      #summation
            self.neural_net[i].A=self.activation(self.neural_net[i].Z,self.neural_net[i].activation)                    #activation
        yhat=self.neural_net[i].A       #output from the last(output) layer
        yhat=np.squeeze(yhat)
        return yhat

    #Perform gradient descent for backward propagation
    def gradient_descent(self, X, Y,max_epoch=10):
        self.learning_rate=0.01                 #assign learning rate as 0.01
        self.max_epoch=max_epoch                #max number of iterations

        for epoch in range(1, max_epoch+1):
            y_hat_train=self.predict(X)                  #update y_hat
            self.backward_propagation(Y,y_hat_train)     #find (dW,db)
            #update the weights and biases of all the layers
            for i in range(1,len(self.layers)):
                self.neural_net[i].W=self.neural_net[i].W-self.learning_rate * self.neural_net[i].dW   #update weights of the respective layer
                self.neural_net[i].b=self.neural_net[i].b-self.learning_rate * self.neural_net[i].db    #update biases of the respective layer

    #This function performs backward propagation by finding dW and db and the mse for the last layer.
    def backward_propagation(self, Y, y_hat):
        for i in range(len(self.layers)-1,0,-1):
            if i+1==len(self.layers):
                dZ=y_hat-Y
            else:
                dZ=np.multiply(np.dot(self.neural_net[i+1].W.T, dZ), self.deactivation(self.neural_net[i].A, self.neural_net[i].activation))
            dW=np.dot(dZ,self.neural_net[i-1].A.T)/(int(0.7*self.x_cols))
            db=np.sum(dZ,axis=1,keepdims=True)/(int(0.7*self.x_cols))
            self.neural_net[i].dW=dW
            self.neural_net[i].db=db

    #Function that trains the neural network by taking x_train and y_train samples as input
    def train(self, X, Y, iterations=500):
        self.fit(X, Y)                          #initialize all the parameters
        if self.check_overfit()==-1:            #check for overfit
            raise Exception('It will overfit')
        for i in range(iterations):
            yhat=self.predict(X)                    #forward propagation
            self.gradient_descent(X,Y,max_epoch=50) #backward propagation
        return yhat         #return the output of the training model
                

#create an object for the neural network model class	
ob1=NN()
yhat=ob1.train(xTrain, yTrain)

#Function that calculates the confusion matrix
def CM(y_test,y_test_obs):
        '''
                Prints confusion matrix
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model
        '''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
                
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        if not (tp==0 and fp==0):
            p= tp/(tp+fp)
        if not (tp==0 and fn==0):
            r=tp/(tp+fn)
        if not (p==0 and r==0):
            f1=(2*p*r)/(p+r)
        if not (tp==0 and tn==0 and fp==0 and fn==0):
            acc=(tp+tn)/(tp+tn+fp+fn)
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy : {acc}")

#check the accuracy of the training dataset
print('\nTraining dataset accuracy: ')
y_train=yTrain[0]
CM(y_train,yhat)

#test the model with the test dataset
y_pred=ob1.predict(xTest)
y_test=yTest[0]

#check the accuracy of the test dataset
print('\nTesting dataset accuracy: ')
CM(y_test,y_pred)


