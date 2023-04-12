# Readme

#### The first effort was [Reproducibility file](https://github.com/RachelRamirez/misclassification_matrix/blob/main/Reproducible_Misclassification_Cost_Matrix_Example.ipynb) 

It was to show reproducible results using three identical neural networks using three different cost matrix functions techniques: the normal default one where there are no additional cost-matrix supplied, and then two Keras' users custom code on a Github Issue board using a cost matrix of all ones.  Once seeded the results between all three were identical, you just have to remember to compare them after restarting and running through once or comparing the correct order.

Once I knew the three methods were equivalent I started to experiment with weights. 

#### The second effort is [w_array[7, 9] = 1.5](https://github.com/RachelRamirez/misclassification_matrix/blob/main/%5B7%2C9%5D_Misclassification_Cost_Matrix_Example.ipynb) 

where I made  the highest misclassification rate according to the confusion matrix on the validation set, have a weighted value of 1.5, assessing results of 30 similarly seeded runs,  which show the variability in the training/validation process, possibly due to the batch order.  I want to make a histogram matrix plot to identify what the distributions look like in each of the confusion matrices. 


However I started to see counterintuitive results, where 7 out of 30 runs the misclassifications would go way up (in the hundreds) if the weight of the mistake was 100.  This led me to experiment with negative numbers and I saw a reverse of the math.  My guess is the method I was using, which i call the isantaro method, had incorrect code somewhere.  So I switched to a different method, which i call the PA method, with code supplied by Phil Alton.  All those files are labelled PA_  and can't be compared outside of them.   I also realized that the number of misclassiications was much larger when I  used Model Shuffle=False.   I thought perhaps the training/validation training and loss history was too jumpy because the batches weren't well mixed so it may not be seeing a fair share of '7's each epoch and leading to epochs where there was higher and lower loss just because of the not as mixed batches.   Model Shuffle=True was used for the rest of the exprments, and you can tell those files are labelled _Shfl_.


#### After discovering the error, around 2/21/2023 I started experimenting with PA Code

This code seems mostly correct.

#### Developing a Baseline PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb

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

The Results of running the "PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb" 30 times (within a for-loop, not restarting the console 30 times) , the greatest number of misclassifications were:


| Actual | Prediction | Avg Misclassified # of Times | Avg Percentage of Times |  Comment | 
| ------ | ---------- | ------------------------ | ------------------- | --- | 
| **9**  | **4**          | 16.466                   | 1.67 %              |  <-- Chosen Baseline |
| 4      | 9          | 15.7                     | 1.55 %              | | 
| 7      | 2          | 12.4                     | 1.20 %              | | 



So for those 30 runs, 9 was misclassified as a 4 the most, with an average of 16 times, per run.   For example on Run 30, training concluded at Epoch 22, and the number of times 9 was misclassified as a 4 was '23'.  The number of times 9 was classified correctly as a 9 was 951.   On Run  0, training concluded at Epoch 17, and 9 was misclassified as a 4, 11 times, and 9 was correctly classified as a 9 958 times.  Run 0 also has 4s misclassified as a 9, 14 times.  So technically in Run 0, after training, the number of misclassifications for 9s as 4s was lower than another misclassification.  Still on average, 9s were mislassified the most.  


### Choosing to `Try to Control 9T-4P' with Lambda

Since we found the highest average number of misclassifications using our 'lousy neural network' was 9_4, we launch into a new set of "PreExperiments" with Lambda using the Phil-Alton code (PA) to try to control it with different weights on the cost matrix.     At this point I have a lot of files named PA_Shfl_... but I choose to focus on just the ones that also have a 40D to signify the 'lousy neural network' to make sure I'm comparing apples with apples.  Of note, it looks like I still was using the original Train/Test Split, not a validation split yet.


- PA_Shfl_w[9,4]_1_40D_Misclassification_Cost_Matrix_Example.ipynb
- - Average 9T_4P: 13.833333333333334

- PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb

- PA_Shfl_w[9,4]_2.0_40D_Misclassification_Cost_Matrix_Example.ipynb
- - Average 9T_4P: 14.5

- PA_Shfl_w[9,4]_10.0_40D_Misclassification_Cost_Matrix_Example.ipynb (Mislabelled, actually was 100, not 10)
- -  Output:    10CM_w[9,4]_PA_100.0_Shfl_40D__2023_02_21_.csv  of shape  (10, 100)
- - Average 9T_4P: 6.4

- PA_Shfl_w[9,4]_1000_40D_Misclassification_Cost_Matrix_Example.ipynb 
- - Average: 0.0

![Graph Builder - Weights 1 10 1000 1000 - Misclass 9T_4P vs logX - Predicted Equation -](https://user-images.githubusercontent.com/13596380/231467943-74e19f49-73a9-4a9e-8e66-ee597a979db7.png)

The above graph is of points X =  1, 2, 10, 100, 1000, and shows a decreasing linear relationship between misclasses of 9T_4P with a log(X), with a lot of variability. 


![image](https://user-images.githubusercontent.com/13596380/231473383-5e53db85-25c7-43b9-b13a-517b64575ec4.png)

The above three sub-graphs show different relationships between the number of misclassifications and the accuracy.  We can see that with one-lambda value we can lower the misclassifications at some cost to overall accuracy.   What I may have needed to do here, is show more Training/Testing Loss and Accuracy Graphs lumped together to see how the Lambda Values effect the overall Training/Testing Loss and Accuracy Cycle.  

If i use acceleration/momentum/learning rate it might be harder to effect the weights of the model later on.  


 
#### Choosing to `Try to Control 9T-4P' with Multiple Lambdas and Epochs
By March 10, I had created the file, [PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_Lambda3.ipynb](https://github.com/RachelRamirez/misclassification_matrix/blob/main/PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_Lambda3.ipynb)

This was to test out different lambda values and different epoch-periods.

From what we knew about previous results using just different lambda values, it seemed like using a weight of 1, 100, and 1000 respectively required {~25} {~15*} {~45 epochs with early stopping} when given 45 epochs and a patience of 10 for early stopping (*30 epochs and patience of 3).  So we decided to split this period up into **Three Phases** Training Phase 1: 5 Epochs, Training Phase 2: 5 Epochs, Training Phase 3: 25 Epochs with Early Stopping. and Patience of 0.

 


<img src = "https://user-images.githubusercontent.com/13596380/231316234-047d6483-cfd2-4f11-bad6-34ed64688bdc.png" alt="Table of Values" width="50%" title="Table of Misclass Values for Different Lambda Values>
 
 
<img src = 
"https://user-images.githubusercontent.com/13596380/231319406-9ae3be38-18df-4db9-bb32-813db7ed1d43.png" alt="Colored First Phase Image" width="70%" height="70%" title="Image Title">

However at this point, it looks like only the initial-phase does anything.  ![image](https://user-images.githubusercontent.com/13596380/231481503-e51cf4a7-403d-4b20-b3bc-dcd789a508b1.png)
