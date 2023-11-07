import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_xy(df):
    # replace nan with -99
    df = df.fillna(-99)
    target = "Loan Status"
    
    x_df = df.drop(target, axis=1)
    y_df = df[target]
    
    y_df = LabelEncoder().fit_transform(y_df)
   
    return x_df, y_df

def run_test(x_df, y_df):
    mlp = MLPClassifier(max_iter=1200)
    ## Feel free to play with these parameters if you want
    #'hidden_layer_sizes': [(5,10), (12), (2,5,10, 15)]
    parameter_space = {
        'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0.01],
        'learning_rate': ['constant','adaptive'],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
    clf.fit(x_df, y_df)

    print('Best parameters found:\n', clf.best_params_)

    #Compare actuals with predicted values 
    y_true, y_pred = y_df , clf.predict(x_df)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))

def test_model(datapath):
    df_base = pd.read_csv(datapath, sep=',')
    x_df, y_df = load_xy(df_base)
    run_test(x_df, y_df)