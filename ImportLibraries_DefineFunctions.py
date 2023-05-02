import time
time.localtime(time.time())
#For Reproducibility
import numpy as np
# np.random.seed(1337)  # for reproducibility

import tensorflow as tf
# tf.random.set_seed(33)

import random as python_random
# python_random.seed(4)

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed
seed = 342
tf.keras.utils.set_random_seed(seed) #Possibly use next iteration if the above doesn't work   #This makes everything VERY DETERMINISTIC


# Running more than once causes variation.  try adding this:
# Set seed value
seed_value = 56
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

print("TF version: " , tf.__version__ )
print("Keras version: " , tf.keras.__version__ )


 

# from __future__ import print_function  #do i still need this?
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import keras.backend as K
from itertools import product
import functools
from functools import partial
from time import ctime
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.metrics import confusion_matrix, classification_report   #CLASSIFICATION REPORT IS FOR F1 SCORE AND BREAKOUT OF AL CATEGORIES ACCURACYS
from  sklearn.utils import shuffle
from sklearn.metrics import f1_score
 
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

print("Finished Loading Libraries")


batch_size = 256 

# I originally had it very  high batch size to reduce the variation in the data each batch and hope 
# it makes the model training more nearly identical which it did, then i bring it back down to something reasonable to get better results training the NN

nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Create a Validation Set
X_val = X_test[:7500]   #take the first 7500 for validation
Y_val = Y_test[:7500]   #Take the first 7500 for validation
y_val = y_test[:7500]

X_test = X_test[7500:]  #Keep the last 2500 for test/holdout
Y_test = Y_test[7500:]  #Keep the last 2500 for test/holdout
y_test = y_test[7500:]

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

#Are the sets relatively balanced? Yes each category is between 8% and 11% per category
print('Train', Y_train.sum(axis=0)/X_train.shape[0])
print('Train # of 9s', Y_train.sum(axis=0)[9])
print('Train # of 4s', Y_train.sum(axis=0)[4])

print('Val', Y_val.sum(axis=0)/X_val.shape[0])
print('Val # of 9s', Y_val.sum(axis=0)[9])
print('Val # of 4s', Y_val.sum(axis=0)[4])

print('Test', Y_test.sum(axis=0)/X_test.shape[0])
print('Test  # of 9s', Y_test.sum(axis=0)[9])
print('Test  # of 4s', Y_test.sum(axis=0)[4])

#@title
class WeightedCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):

  def __init__(self, cost_mat, name='weighted_categorical_crossentropy', **kwargs):

    cost_mat = np.array(cost_mat)   
    ## when loading from config, self.cost_mat returns as a list, rather than an numpy array. 
    ## Adding the above line fixes this issue, enabling .ndim to call sucessfully. 
    ## However, this is probably not the best implementation
    assert(cost_mat.ndim == 2)
    assert(cost_mat.shape[0] == cost_mat.shape[1])
    super().__init__(name=name, **kwargs)
    self.cost_mat = K.cast_to_floatx(cost_mat)

  def __call__(self, y_true, y_pred, sample_weight=None):
    assert sample_weight is None, "should only be derived from the cost matrix"  
    return super().__call__(
        y_true=y_true, 
        y_pred=y_pred, 
        sample_weight=get_sample_weights(y_true, y_pred, self.cost_mat),
    )


  def get_config(self):
    config = super().get_config().copy()
    # Calling .update on the line above, during assignment, causes an error with config becoming None-type.
    config.update({'cost_mat': (self.cost_mat)})
    return config

  @classmethod
  def from_config(cls, config):
    # something goes wrong here and changes self.cost_mat to a list variable.
    # See above for temporary fix
    return cls(**config)

def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)

    y_pred.shape.assert_has_rank(2)
    assert(y_pred.shape[1] == num_classes)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred = K.one_hot(K.argmax(y_pred), num_classes)

    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n


# Register the loss in the Keras namespace to enable loading of the custom object.
tf.keras.losses.WeightedCategoricalCrossentropy = WeightedCategoricalCrossentropy
 
#@title
def plot_model_history(model_history, nb_epoch, cm3): 
  # Parameters
  # ----------
  # model_history : keras.callbacks.History
  #     The history object returned by the fit() method of the model.
  # cm3 : 10x10 dataframe 
  #      10x10 dataframe of confusion matrix from predicted X_val categories
  # nb_epoch = restored_weights : int
  #     The epoch at which the weights were restored.
  # tot_epochs : int
  #     Calculated Total number of epochs for which the model was trained.
  
   
  tot_epochs = max(model_history.epoch)+1  #if the total epochs ran is 28, it'll show up as 27 in the epoch object so we must add 1
  #print("Total Epochs: ", tot_epochs)

  #if tot_epochs is the total number of epochs ran then early stop did not happen, and we need not minus patience
  if tot_epochs == nb_epoch:
    restored_weights = tot_epochs
  else:
    restored_weights  = tot_epochs-patience   #when using restore-best-weights and patience, it'll restore the best weights back
  #print("Restored weights at ", restored_weights, "Patience used: ", patience)

  fig = plt.figure(figsize=(20, 10))
  fig, ax = plt.subplots(1,3)
  ax[0].plot(range(1,tot_epochs+1), model_history.history['categorical_accuracy'], color='blue',             label='Training')
  ax[0].plot(range(1,tot_epochs+1), model_history.history['val_categorical_accuracy'] , color='orange',             label='Validation')
  ax[0].scatter((restored_weights), model_history.history['val_categorical_accuracy'][restored_weights-1] , color='orange')
  ax[0].scatter(restored_weights, model_history.history['categorical_accuracy'][restored_weights-1], color='blue')
  ax[0].annotate(text=str(restored_weights),  xy=(restored_weights, model_history.history['val_categorical_accuracy'][restored_weights-1]),
                  textcoords="offset points", xytext=(0,10), ha='center', color='black')
  ax[0].legend()
  ax[0].set_title('Training and Validation Accuracy')

  ax[1].plot(range(1,tot_epochs+1), model_history.history['loss'], color= 'blue', label='Training')
  ax[1].plot(range(1,tot_epochs+1), model_history.history['val_loss'], color='orange', label='Validation')
  ax[1].scatter(restored_weights, model_history.history['loss'][restored_weights-1], color='blue')
  ax[1].scatter((restored_weights), model_history.history['val_loss'][restored_weights-1] , color='orange')
  ax[1].annotate(text=str(restored_weights),  xy=(restored_weights, model_history.history['val_loss'][restored_weights-1]),
                  textcoords="offset points", xytext=(0,10), ha='center')
  ax[1].legend()
  ax[1].set_title('Training and Validation Loss')


  cm3_wodiag = cm3*(np.ones((10,10)) - np.eye(10))

  ax[2] = sns.heatmap(cm3_wodiag, annot=True, annot_kws={"size": 7},  fmt='g', cmap=sns.cm.rocket_r) # font size
  ax[2].set_xlabel('Predicted Class')
  ax[2].set_ylabel('True Class')
  ax[2].set_title('# of misclassifications of 9 as 4 is '+str(cm3[4][9]))
  cbar = ax[2].collections[0].colorbar
  cbar.remove() # Just takes up valuable room and is worthless


  plt.gcf().set_size_inches(15, 5)  # this works 
  # plt.gcf().suptitle(f"Lambda Value {lambda_val} for {nb_epoch} Epochs and Patience {patience} " )


  #@title
def create_model(): #Removed cost-matrix which is called up in the Compile Function and passed to the weighted-loss function
  model = Sequential()
  model.add(Dense(40, input_shape=(784,), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(40, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(10,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model.add(Activation('softmax'))
  return model  #I removed Compile



  #@title
def log_confusion_matrix( epoch, logs):
  # Use the model to predict the values from the validation dataset.
  y_prediction = model.predict(X_val, verbose=0)     #I call it y_prediction3 because I just want to make sure this is  updated within and not interfering with the other prediction below
  y_prediction  = np.argmax(y_prediction, axis=1)

  #Create confusion matrix 
  cm = confusion_matrix(y_val, y_prediction)
  cm_array = np.asarray(cm)  #Indiv CM as array for storing
  logs['9T_4P'] = cm[9,4]
  logs['4T_9P'] = cm[4,9]

  logs['0T_Acc'] = cm[0,0]/np.sum(cm[0])
  logs['1T_Acc'] = cm[1,1]/np.sum(cm[1])
  logs['2T_Acc'] = cm[2,2]/np.sum(cm[2])
  logs['3T_Acc'] = cm[3,3]/np.sum(cm[3])
  logs['4T_Acc'] = cm[4,4]/np.sum(cm[4])
  logs['5T_Acc'] = cm[5,5]/np.sum(cm[5])
  logs['6T_Acc'] = cm[6,6]/np.sum(cm[6])
  logs['7T_Acc'] = cm[7,7]/np.sum(cm[7])
  logs['8T_Acc'] = cm[8,8]/np.sum(cm[8])
  logs['9T_Acc'] = cm[9,9]/np.sum(cm[9])


  logs['cm_per_epoch'] = cm_array.reshape((1,100))

#@title
def log_classification_report( epoch, logs):
  # Use the model to predict the values from the validation dataset.
    y_prediction = model.predict(X_val, verbose=0)     #I call it y_prediction3 because I just want to make sure this is  updated within and not interfering with the other prediction below
    y_prediction  = np.argmax(y_prediction, axis=1)

    #Create confusion matrix 
    cr = classification_report(y_val, y_prediction)
    # print(cr)
    logs['cr_per_epoch'] = cr 


    #@title
def log_f1_score( epoch, logs):
  # Use the model to predict the values from the validation dataset.
    y_prediction = model.predict(X_val, verbose=0)     #I call it y_prediction3 because I just want to make sure this is  updated within and not interfering with the other prediction below
    y_prediction  = np.argmax(y_prediction, axis=1)


    logs["f1_micro"] = f1_score(y_val, y_prediction, average='micro')
    logs["f1_macro"] = f1_score(y_val, y_prediction, average='macro')
    logs["f1_weighted"] = f1_score(y_val, y_prediction, average='weighted')
    logs["f1_notweighted"] = f1_score(y_val, y_prediction, average=None)


#@title
def return_cm(model):
  y_prediction = model.predict(X_val, verbose=0)
  y_prediction  = np.argmax(y_prediction, axis=1)
  # Y_prediction = np_utils.to_categorical(y_prediction, nb_classes)

  cm3 = confusion_matrix(y_val, y_prediction)
  cm3 = pd.DataFrame(cm3, range(10),range(10))
  return cm3

  # # plt.figure(figsize = (4,4))
  # # cm3
  # sns.heatmap(cm3, annot=True, annot_kws={"size": 7},  fmt='g') # font size
  # plt.show()
  # # cm_using_weighted_new = cm3
 #@title

def return_cr(model):
  y_prediction = model.predict(X_val, verbose=0)
  y_prediction  = np.argmax(y_prediction, axis=1)
  # Y_prediction = np_utils.to_categorical(y_prediction, nb_classes)

  cr = classification_report(y_val, y_prediction)
  print(cr)
  return cr


def return_f1score(model):
  # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')[source] 
  y_prediction = model.predict(X_val, verbose=0)
  y_prediction  = np.argmax(y_prediction, axis=1)
  # Y_prediction = np_utils.to_categorical(y_prediction, nb_classes)

  f1_notweighted = f1_score(y_val, y_prediction, average=None)
  print(f1_notweighted)
  f1_micro = f1_score(y_val, y_prediction, average='micro')
  f1_macro = f1_score(y_val, y_prediction, average='macro')
  f1_weighted = f1_score(y_val, y_prediction, average='weighted')
  # print("Micro: ", f1_micro, "Macro: ", f1_macro, "Weighted: ", f1_weighted)
  print("Micro: {:.5f} Macro: {:.5f} Weighted: {:.5f}".format(f1_micro, f1_macro, f1_weighted))
  # return f1



#@title: plot_model_history_all
def plot_model_history_all(model_history, nb_epoch=None, cm3=None): 
  # Parameters
  # ----------
  # tot_epochs : int
  #     Total number of epochs for which the model was trained.
  # model_history : keras.callbacks.History
  #     The history object returned by the fit() method of the model.
  # cm3 : 10x10 dataframe 
  #      10x10 dataframe of confusion matrix from predicted X_val categories
  # restored_weights : int
  #     The epoch at which the weights were restored.

  
  fig = plt.figure(figsize=(20, 10))
  fig, ax = plt.subplots(1,2)
  
  tot_epochs = max(model_history.epoch)+1  #if the total epochs ran is 28, it'll show up as 27 in the epoch object so we must add 1
  # print("Total Epochs: ", tot_epochs)

  #if tot_epochs is the total number of epochs ran then early stop did not happen, and we need not minus patience
  if tot_epochs == nb_epoch:
    restored_weights = tot_epochs
  else:
    restored_weights  = tot_epochs-patience   #when using restore-best-weights and patience, it'll restore the best weights back
  # print("Restored weights at ", restored_weights, "Patience used: ", patience)

  ax[0].plot(range(1,tot_epochs+1), model_history.history['categorical_accuracy'], color='blue',           )
  ax[0].plot(range(1,tot_epochs+1), model_history.history['val_categorical_accuracy'] , color='orange',    )
  ax[0].scatter((restored_weights), model_history.history['val_categorical_accuracy'][restored_weights-1] , color='orange')
  ax[0].scatter(restored_weights, model_history.history['categorical_accuracy'][restored_weights-1], color='blue')
  ax[0].annotate(text=str(restored_weights),  xy=(restored_weights, model_history.history['val_categorical_accuracy'][restored_weights-1]),
                  textcoords="offset points", xytext=(0,10), ha='center', color='black')
  # ax[0].legend()
  ax[0].set_title('Training (Blue) and Validation (Orange) Accuracy', fontsize='8')

  ax[1].plot(range(1,tot_epochs+1), model_history.history['loss'], color= 'blue',  )
  ax[1].plot(range(1,tot_epochs+1), model_history.history['val_loss'], color='orange',  )
  ax[1].scatter(restored_weights, model_history.history['loss'][restored_weights-1], color='blue')
  ax[1].scatter((restored_weights), model_history.history['val_loss'][restored_weights-1] , color='orange')
  ax[1].annotate(text=str(restored_weights),  xy=(restored_weights, model_history.history['val_loss'][restored_weights-1]),
                  textcoords="offset points", xytext=(0,10), ha='center')
  # ax[1].legend()
  ax[1].set_title('Training (Blue) and Validation (Orange) Loss' , fontsize='8')


  plt.gcf().set_size_inches(10, 5)  # this works 
  # plt.gcf().suptitle(f"Lambda Value {lambda_val} for {nb_epoch} Epochs and Patience {patience} " )

  








## ----------------------------------------------------------------------------------------------
## Tried to create a Callback to call the Classifcatinreport ever 5 epochs, but it doesnt work 
## because it cant see the model.  ZGoing back to trying to define the function to only be called every 5th epoch


# class cr_callback(tf.keras.callbacks.Callback):

#     def on_epoch_end(self, epoch, log=None):

#         if epoch % 5 == 0:  # <- add additional condition here
#             self._do_the_stuff()
            
            
#     def _do_the_stuff(self, model):
#         print('Do the stuff')
#         y_prediction = model.predict(X_val, verbose=0)
#         y_prediction  = np.argmax(y_prediction, axis=1)
#         # Y_prediction = np_utils.to_categorical(y_prediction, nb_classes)
#         cr = classification_report(y_val, y_prediction)
#         logs['cr'] = cr

#           # return cr
        
#     # def on_training_end(self, logs=None):
#     #     self._do_the_stuff()



## Saw this example ono Kaggle, but couldn't get it to work, it does give me an idea(if epoch==0)

# from sklearn.metrics import accuracy_score
# class every10epochCallback(tf.keras.callbacks.Callback):
#     def __init__(self, X_val, Y_val):
#         super().__init__()
#         self.X = X_val
#         self.y = Y_val.argmax(axis=1)
#     def on_epoch_begin(self, epoch, logs=None):
#         if epoch == 0:
#             return
#         if epoch%10==0:
#             pred = (model.predict(self.X))
#             print('epoch: ',epoch,'  ,Accuracy:  ',accuracy_score(self.y,pred.argmax(axis=1)),' ')

# model.fit(X_train, Y_train,batch_size=batch_size,epochs=10,verbose=0,  validation_data=(X_val, Y_val), shuffle=True, use_multiprocessing=True, 
#           callbacks=[every10epochCallback(X_val,Y_val)])
