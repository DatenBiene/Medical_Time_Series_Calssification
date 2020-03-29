from builtins import print
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

import os
import operator

import utils

from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.interpolate import interp1d


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(path,NAME):

    if NAME == 'MIT-BIH':
        data = pd.read_csv(path,header=None)
        labels = data[187].astype(int)
        data.drop(187,axis=1,inplace=True)
        data = data.values
        labels = labels.values

    elif NAME == 'ECG5000':
        data = []
        labels = []
        with open(path) as file:
            for row in file:
                series = row.strip('\n')
                series = series.split()
                labels.append(int(float(series[0])))
                ecg = [float(s) for s in series[1:]]
                data.append(ecg)
        data,labels = np.array(data),np.array(labels,dtype=int)

    elif NAME == 'transplant':
        x_train, y_train, x_test, y_test, labels = np.load(path)
        return x_train,y_train,x_test,y_test,labels
    else:
        raise ValueError(f'Name {NAME} not recognized')

    return data,labels


def split_dataset(data,labels,validation=True , val_prop = 0.2):
    """
    splits dataset into (train,test) or (train,validation,test)
    arguments
    ---------
    data: features (X), array-like
    labels : classes (y), array-like
    validation: bool, if True split in (train,validation,test) with proportion (0.6,0.2,0.2)
                      if False split in (train,test) with proportion (0.8,0.2)

    returns
    -------
    xtrain,xtest,(xval): splitted versions of data
    ytrain,ytest,(yval): splitted versions of labels as one hot encoding if more than 2 classes
    y_test_true,(y_val_true): not one hot encoded versions of labels
    """
    labels = np.array(labels,dtype=int)
    data = np.array(data)

    if validation:
        val_proportion = val_prop*2
    else:
        val_proportion = val_prop
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_proportion, random_state=0)
    #split train test
    for train_index,test_index  in sss.split(data,labels):
        xtrain,xtest = data[train_index],data[test_index]
        ytrain,ytest = labels[train_index],labels[test_index]

    if validation:
        #split validation test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for val_index,test_index in sss.split(xtest,ytest):
            xval,xtest = xtest[val_index],xtest[test_index]
            yval,ytest = ytest[val_index],ytest[test_index]

    if len(np.unique(labels))>2:
        ytrain = to_categorical(ytrain)
        y_test_true = ytest #not categorical
        ytest = to_categorical(ytest)

        if validation:
            y_val_true = yval #not categorical
            yval = to_categorical(yval)
    else:
        if validation:
            y_val_true = yval
        y_test_true = ytest

    if validation:
        return xtrain,ytrain,xval,yval,y_val_true,xtest,ytest,y_test_true
    else:
        return xtrain,ytrain,xtest,ytest,y_test_true


def visualize_transplant(X,y,cols):
    fig = plt.figure(figsize=(14,15))
    series0 = X[y==0][0]
    series1 = X[y==1][0]
    N = series0.shape[1]
    for i in range(N):
        ax0 = fig.add_subplot(N,2,2*i+1)
        ax1 = fig.add_subplot(N,2,2*i+2)

        ax0.plot(series0[:,i])
        ax1.plot(series1[:,i])

        ax0.set_ylabel(cols[i],fontsize=16)
        if i == 0:
            ax0.set_title('Example of label 0',fontsize=16)
            ax1.set_title('Example of label 1',fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_mitbih(X,y):
    fig = plt.figure(figsize=(14,5))
    for c in set(y):
        spe_ecg = X[y==c][0]
        plt.plot(spe_ecg,label='example of label '+str(c))

    plt.legend()
    plt.show()

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res



def plot_epochs_metric(hist, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
