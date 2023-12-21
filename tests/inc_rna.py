##include rna 
## contient les fonctions pour l'utilisations des rna 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import utility as u
from sklearn.model_selection import train_test_split


#3_2_1

class NeuralNet:
    def __init__(self,X_train=None,y_train = None,X_test = None,y_test = None,
        hidden_layer_sizes=(14,),activation = 'identity',learning_rate=0.1,epoch=200):
        self._nlayers = len(hidden_layer_sizes)
        self._ninput = 0
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._nhiddenlayers = 0
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._learnnng_rate = learning_rate
        self._epoch = epoch

        self._weights_train = [None] * (self._nlayers+1)
        self._weights_test = [None] * (self._nlayers+1)
        self._biaises = [None] * (self._nlayers+1)

        self._errors = [None] * (epoch)

        self._Z = [None] * (self._nlayers+1) 
        self._A = [None] * (self._nlayers+1) 
        self._df = [None] * (self._nlayers+1) 

        self._delta = [None] * (self._nlayers + 1)
        self._dW    = [None] * (self._nlayers + 1)
        self._db    = [None] * (self._nlayers + 1)
        
        self._weights_initialization(X_train, y_train)
        self._weights_initialization(X_test,y_test)
        return 

    def _weights_initialization(self, X, y):
        np.random.seed(42)
        n_cols=X.shape[1]
        self.n_inputs=X.shape[1]
        for i in range(0,self.n_layers):
            n_lines= self.hidden_layer_sizes[i]
            self.weights[i]=np.random.uniform(-1,1,(n_lines,n_cols))
            self.biases[i]=np.random.uniform(-1,1,(n_lines,1))
            n_cols=n_lines                   
        l_out=self.n_layers
        n_lines=y.shape[1]
        self.weights[l_out]=np.random.uniform(-1,1,(n_lines,n_cols))
        self.biases[l_out]=np.random.uniform(-1,1,(n_lines,1))
        return

    def forward_propagation(self,X,Y):
        U = u.Utility
        activation = self._activation
        if activation == "identity":
            self._Z,self._A = U.identity(self._Z) 
        elif activation == "tanh":
            self._A,self._df = U.tanh(self._Z)
        elif activation == "relu":
            self._A,self._df = U.relu(self._Z)
        elif activation == "sigmoid":
            self._A,self._df = U.sigmoid(self._Z)
        for i in range(0,(self._nlayers_input)):
            self._Z[i+1] = self._weights[i+1]*self._A[i] + self._biaises[i+1]
            self._A[i+1] = np.diff(self._Z[i+1])
        y_hat = np.diff(U.softmax(self._Z))
        self._A[i] = y_hat
        print(U.cross_entropy_cost(y_hat,0))


#3_2_3

#Part2

    def backward_pass(self,X,y):
        delta = [None] * (self._nlayers_input + 1)
        dW = [None] * (self._nlayers_input + 1)
        db = [None] * (self._nlayers_input + 1 ) 
    


#2
#Operation necessaire pour eviter le sur-apprentissage

    def train_epoch(self):
        err_train = []
        i = 0
        for X in self._X_train:
            X = np.transpose(X)
            err_train[i] = self.forward_propagation(X,None)
            self.backward_pass(X)
            i += 1
        err_train = np.mean(err_train)


        err_test = []
        j = 0
        for X in self.X_test:
            X = X.transpose
            err_test[j] = self.forward_propagation(X)
            j += 1
        err_test = np.mean(err_test)
        return err_train,err_test
#3




