# Readme

#### The first effort is [Reproducibility file](https://github.com/RachelRamirez/misclassification_matrix/blob/main/Reproducible_Misclassification_Cost_Matrix_Example.ipynb) was to show reproducible results using three identical neural networks using three different cost matrix functions techniques: the normal default one where there are no additional cost-matrix supplied, and then two Keras' users custom code on a Github Issue board using a cost matrix of all ones.  Once seeded the results between all three were identical, you just have to remember to compare them after restarting and running through once or comparing the correct order.

Once I knew the three methods were equivalent I started to experiment with weights. 

#### The second effort is [w_array[7, 9] = 1.5](https://github.com/RachelRamirez/misclassification_matrix/blob/main/%5B7%2C9%5D_Misclassification_Cost_Matrix_Example.ipynb) where I made  the highest misclassification rate according to the confusion matrix on the validation set, have a weighted value of 1.5.  This is where I am now, assessing results of 30 similarly seeded runs,  which show the variability in the training/validation process, possibly due to the batch order.  I want to make a histogram matrix plot to identify what the distributions look like in each of the confusion matrices. 


However I started to see counterintuitive results, where 7 out of 30 runs the misclassifications would go way up (in the hundreds) if the weight of the mistake was 100.  This led me to experiment with negative numbers and I saw a reverse of the math.  My guess is the method I was using, which i call the isantaro method, had incorrect code somewhere.  So I switched to a different method, which i call the PA method, with code supplied by Phil Alton.  All those files are labelled PA_  and can't be compared outside of them.   I also realized that the number of misclassiications was much larger when I  used Model Shuffle=False.   I thought perhaps the training/validation training and loss history was too jumpy because the batches weren't well mixed so it may not be seeing a fair share of '7's each epoch and leading to epochs where there was higher and lower loss just because of the not as mixed batches.   Model Shuffle=True was used for the rest of the exprments, and you can tell those files are labelled _Shfl_.

#### After discovering the error, around 2/21/2023 I started experimenting with PA Code

This code seems mostly correct.

### PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb

With this file i redux the neural network model with two-layers of 40 Dense Connections and two dropout between.  I gave the misclassification weights all 1's, so this file serves as a Baseline.   Importantly up to this point I had been using the Default MNIST Train/Test Set split, and used the Test Set directly as my Validation set.  At this point I had the following hyperparameters/code:

```
batch size: 256
number of epochs: 30
early patience: 3 on validation loss 

model:
def PA_method(cost_matrix):
  model3 = Sequential()
  model3.add(Dense(40, input_shape=(784,), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model3.add(Activation('relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(40, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model3.add(Activation('relu'))
  model3.add(Dropout(0.2))
  model3.add(Dense(10,kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42)))
  model3.add(Activation('softmax'))

  rms = RMSprop()

  model3.compile(loss=WeightedCategoricalCrossentropy(cost_matrix), optimizer=rms,  metrics='categorical_accuracy',)
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

  model3_history = model3.fit(X_train, Y_train,
            batch_size=batch_size, epochs=nb_epoch, verbose=2,
            validation_data=(X_test, Y_test), shuffle=True, use_multiprocessing=True
            ,callbacks = [callback]
            )

 

  #Predict
  y_prediction = model3.predict(X_test)
  y_prediction  = np.argmax(y_prediction, axis=1)
  
  #Confusion Matrix
  cm3 = confusion_matrix(y_test, y_prediction)
  cm3 = pd.DataFrame(cm3, range(10),range(10))
  return cm3
```

The Results of running the "PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb" 30 times (within a for-loop, not restarting the console 30 times) were the greatest number of misclassifications were:


| Actual | Prediction | Misclassified # of Times | Percentage of Times |
| ------ | ---------- | ------------------------ | ------------------- |
| 9      | 4          | 16.466                   | 1.67 %              |
| 4      | 9          | 15.7                     | 1.55 %              |
| 7      | 2          | 12.4                     | 1.20 %              |



