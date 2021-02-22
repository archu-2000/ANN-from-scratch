#code for preprocessing the dataset
#import all the required libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

#function to read the dataset and preprocess it
def read_dataset():
    df=pd.read_csv(r'LBW_Dataset.csv')
    df.fillna(df.mean(), inplace=True)      #fill the null spaces with the mean of the corresponding column
    scaler=preprocessing.StandardScaler()       #to standardize the values of the dataset
    scaled_data=scaler.fit_transform(df.iloc[:,:-1])
    df1=pd.DataFrame(scaled_data)           
    frames=[df1,df.iloc[:,-1]]
    df2=pd.concat(frames,axis=1)        #concatenate the standardized input features and the target from the original dataset
    df2.to_csv('LBW_preprocessed.csv', index=False)  #write into a new file
    
read_dataset()  #call the function
