import math
from math import sqrt
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


train_data=pd.read_csv("train_time_series.csv")
train_label=pd.read_csv("train_labels.csv")
test_data=pd.read_csv("test_time_series.csv")
test_label=pd.read_csv("test_labels.csv")

x= train_data["x"]
y= train_data["y"]
z= train_data["z"]
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=z)


df=pd.DataFrame(train_label, columns =['timestamp','label'])
df1=pd.DataFrame(test_label, columns =['timestamp','label'])


def find_index_train_label(time):
    for i in range(len(train_data)):
        if time==train_data["timestamp"][i]:
            return i
        
def find_index_test_label(time):
    for i in range(len(test_data)):
        if time==test_data["timestamp"][i]:
            return i


def implement_linear_regression():
    i=0
    j=0
    x_train=np.empty((len(train_label)))
    y_train=np.empty((len(train_label)))
    z_train=np.empty((len(train_label)))
    x_test=np.empty((len(test_label)))
    y_test=np.empty((len(test_label)))
    z_test=np.empty((len(test_label)))
 
    for train_row in train_label["timestamp"]:
        pos_train=find_index_train_label(train_row)
        x_train[i]= train_data['x'][pos_train]
        y_train[i]=train_data['y'][pos_train]
        z_train[i]=train_data['z'][pos_train]
        i+=1
##        df.insert(2,'x',train_data['x'][pos_train],True)
##        df.insert(3,'y',train_data['y'][pos_train],True)
##        df.insert(4,'z',train_data['z'][pos_train],True)
     

    for test_row in test_label["timestamp"]:
        pos_test=find_index_test_label(test_row)
        x_test[j]=test_data['x'][pos_test]
        y_test[j]=test_data['y'][pos_test]
        z_test[j]=test_data['z'][pos_test]
        j+=1
##        
##        
##        df1.insert(4,'z',test_data['z'][pos_test],True)
        
        
    print("h")    
    df1.insert(2,'x',x_test,True)
    df1.insert(3,'y',y_test,True)
    df1.insert(4,'z',z_test,True)
    df.insert(2,'x',x_train,True)
    df.insert(3,'y',y_train,True)
    df.insert(4,'z',z_train,True)
    y_ins=df['label']
    x_ins=df[['x','y','z']]
    linear_regression=LinearRegression()
    linear_regression.fit(x_ins,y_ins)
    X=df1[['x','y','z']]
    y_pred=linear_regression.predict(X)
    print(y_pred)

implement_linear_regression()

