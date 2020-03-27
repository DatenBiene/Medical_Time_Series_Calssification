#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow_addons as tfa
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense, BatchNormalization,BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import save_logs,plot_epochs_metric
from utils.utils import calculate_metrics


def add_conv_block_input(model, n_filters, kern_size, n_stride, fc_act, pooling, coeff_dropout, input_shape):
    model.add(Conv1D(filters = n_filters, kernel_size = kern_size, 
                     strides = n_stride, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(fc_act))
    model.add(Dropout(coeff_dropout))
    model.add(Conv1D(filters = n_filters, kernel_size = kern_size, 
                     strides = n_stride, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(fc_act))
    model.add(Dropout(coeff_dropout))
    model.add(pooling)
    
def add_conv_block(model, n_filters, kern_size, n_stride, fc_act, coeff_dropout, pooling):
    model.add(Conv1D(filters = n_filters, kernel_size = kern_size, 
                     strides = n_stride, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(fc_act))
    model.add(Dropout(coeff_dropout))
    model.add(Conv1D(n_filters, n_stride, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(fc_act))
    model.add(Dropout(coeff_dropout))
    model.add(pooling)

class Classifier_MLP:

    def __init__(self, output_directory, input_shape, nb_classes, hidden_layers_size,verbose=False,build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes,hidden_layers_size)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, n_filters, kern_size, n_stride, input_shape, add_FC, n_unit_FC):

        model = Sequential()

        add_conv_block_input(model, n_filters[0], kern_size[0], n_stride[0], 'relu', MaxPooling1D(3), 0.1, input_shape)
        for k in range(1,n_conv_block):
            add_conv_block(model, n_filters[k], kern_size[k], n_stride[k], 'relu', 0.1, MaxPooling1D(3))

        model.add(Flatten())
        if add_FC:
            model.add(Dense(n_unit_FC))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
    
        model.add(Dense(n_classes, activation='softmax'))
        
        optimizer = SGD(learning_rate=0.05)
        optimizer = tfa.optimizers.SWA(optimizer, start_averaging=0, average_period=3)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.01)

        file_path = self.output_directory+'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val,y_true,batch_size=16,nb_epochs=500):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        if len(y_true.shape)>1:
            y_true = np.argmax(y_true,axis=1)

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        #model = keras.models.load_model(self.output_directory+'best_model.hdf5')

        #y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        #y_pred = np.argmax(y_pred , axis=1)

        plot_epochs_metric(hist,'loss')

        keras.backend.clear_session()
        return hist

    def predict(self, x_test, y_true,return_df_metrics = True):
        if len(y_true.shape)>1:
            y_true = np.argmax(y_true,axis=1)
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred

