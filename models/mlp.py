# MLP model
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import plot_epochs_metric
from utils.utils import calculate_metrics

class Classifier_MLP:

	def __init__(self, output_directory, input_shape, nb_classes, hidden_layers_size,min_lr=0.0001,verbose=False,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes,hidden_layers_size,min_lr)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory + 'model_init.hdf5')

		return

	def build_model(self, input_shape, nb_classes,hidden_layers_size,min_lr):

		if nb_classes==2:
			loss = 'binary_crossentropy'
			n_units_dense = 1
		else:
			loss = 'categorical_crossentropy'
			n_units_dense = nb_classes

		n_layers = len(hidden_layers_size)

		model = keras.models.Sequential()
		model.add(keras.layers.Input(input_shape))

		# flatten/reshape because when multivariate all should be on the same axis
		model.add(keras.layers.Flatten())

		for i in range(n_layers):
			model.add(keras.layers.Dropout(0.1))
			model.add(keras.layers.Dense(hidden_layers_size[i], activation='relu'))

		model.add(keras.layers.Dropout(0.3))
		model.add(keras.layers.Dense(n_units_dense, activation='softmax'))


		model.compile(loss=loss, optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=min_lr)

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
		# x_val and y_val are only used to monitor the test loss and NOT for training

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

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
