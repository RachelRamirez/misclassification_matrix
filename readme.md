# Readme

## Tinkering

**The first effort was [Reproducibility file](https://github.com/RachelRamirez/misclassification_matrix/blob/main/Reproducible_Misclassification_Cost_Matrix_Example.ipynb)** 

It was to show reproducible results using three identical neural networks using three different cost matrix functions techniques: the normal default one where there are no additional cost-matrix supplied, and then two Keras' users custom code on a Github Issue board using a cost matrix of all ones.  Once seeded the results between all three were identical, you just have to remember to compare them after restarting and running through once or comparing the correct order.

Once I knew the three methods were equivalent I started to experiment with weights. 

**The second effort is [w_array[7, 9] = 1.5](https://github.com/RachelRamirez/misclassification_matrix/blob/main/%5B7%2C9%5D_Misclassification_Cost_Matrix_Example.ipynb)**

where I made  the highest misclassification rate according to the confusion matrix on the validation set, have a weighted value of 1.5, assessing results of 30 similarly seeded runs,  which show the variability in the training/validation process, possibly due to the batch order.   


However I started to see **counterintuitive results**, where 7 out of 30 runs the misclassifications would go way up (in the hundreds) if the weight of the mistake was 100.  This led me to experiment with negative numbers and I saw a reverse of the math.  My guess is the method I was using, which i call the isantaro method, had incorrect code somewhere.  So I switched to a different method, which i call the **PA method**, with code supplied by Phil Alton.  All those files are labelled PA_  and can't be compared outside of them.   I also realized that the number of misclassiications was much larger when I  used Model Shuffle=False.   I thought perhaps the training/validation training and loss history was too jumpy because the batches weren't well mixed so it may not be seeing a fair share of '7's each epoch and leading to epochs where there was higher and lower loss just because of the not as mixed batches.   **Model Shuffle=True** was used for the rest of the exprments, and you can tell those files are labelled _Shfl_.


After discovering the error, around 2/21/2023 I started experimenting with PA Code and you can tell those files are also labelled with 'PA_'
His code seems to behave correctly.

------


## PRE-EXPERIMENTS

**Developing a Baseline PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb**


With this file i redux the neural network model with two-layers of 40 Dense Connections to "increase" the number of errors!  I gave the misclassification weights all 1's, so this file serves as a Baseline.   Importantly up to this point I had been using the Default MNIST Train/Test Set split, and used the Test Set directly as my Validation set.  At this point I had the following hyperparameters/code:

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

The fact that one of my runs incorrectly classified a 9 as a 4, 23 times, bothers me, because when I run my multi-lambda-epochs 1-1-1 30 times later, I get a more narrow range, which makes me doubt the results.  

----------

#### Choosing to 'Try to Control 9T-4P' with One Lambda Value (Misclassification Cost on 9T_4P) throughout all the training

Since we found the highest average number of misclassifications using our 'lousy neural network' was 9_4, we launch into a new set of "PreExperiments" with Lambda using the Phil-Alton code (PA) to try to control it with different weights on the cost matrix.     At this point I have a lot of files named PA_Shfl_... but I choose to focus on just the ones that also have a 40D to signify the 'lousy neural network' to make sure I'm comparing apples with apples.  Of note, it looks like I still was using the original Train/Test Split, not a validation split yet.

List of files in Github that have Output easily viewable:
- PA_Shfl_w[9,4]_1_40D_Misclassification_Cost_Matrix_Example.ipynb  (Average 9T_4P: 13.833333333333334)
- PA_Shfl_w[7,2]_1.0_40D_Misclassification_Cost_Matrix_Example.ipynb  
- PA_Shfl_w[9,4]_2.0_40D_Misclassification_Cost_Matrix_Example.ipynb  (Average 9T_4P: 14.5)
- PA_Shfl_w[9,4]_10.0_40D_Misclassification_Cost_Matrix_Example.ipynb (Mislabelled, actually was 100, not 10) (Average 9T_4P: 6.4)
- PA_Shfl_w[9,4]_1000_40D_Misclassification_Cost_Matrix_Example.ipynb (Average 9T_4P: 0.0)


**Log-Linear Trend between Misclassifications and Lambda-Value**

![Graph Builder - Weights 1 10 1000 1000 - Misclass 9T_4P vs logX - Predicted Equation -](https://user-images.githubusercontent.com/13596380/231467943-74e19f49-73a9-4a9e-8e66-ee597a979db7.png)

The above graph is of points X =  1, 2, 10, 100, 1000, and shows a decreasing linear relationship between misclasses of 9T_4P with a log(X), with a lot of variability. 

**Multiple Graphs to show Multiple Relationships with Misclassifications and Accuracy - Still one-constant Lambda-Value**

The following graph was created later for lambda-combos, but was run on the supposed static-lambda-combinations to show reproducibility.  I'm a little concerned that for the 1-1-1 lambda-combo runs, the 9T_4P is only showing a range between 12 and 16, it seems like at least one of them should be bigger?  But maybe that's because it's run on the Validation Set.

![image](https://user-images.githubusercontent.com/13596380/231473383-5e53db85-25c7-43b9-b13a-517b64575ec4.png)

The above three sub-graphs show different relationships between the number of misclassifications and the accuracy.  We can see that with one-lambda value we can lower the misclassifications at some cost to overall accuracy.   What I may have needed to do here, is show more Training/Testing Loss and Accuracy Graphs lumped together to see how the Lambda Values effect the overall Training/Testing Loss and Accuracy Cycle.  

If i use acceleration/momentum/learning rate it might be harder to effect the weights of the model later on.  


-----------

### PreExperimentation Part 2: Choosing to 'Try to Control 9T-4P' with Multiple Lambdas over THREE Training Phases

By March 10, I had created the file, [PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_Lambda3.ipynb](https://github.com/RachelRamirez/misclassification_matrix/blob/main/PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_Lambda3.ipynb) and importantly I at least have evidence of splitting the train set into a validation set so that there are:

- 60000 train samples
- 7500 validation samples
- 2500 test samples

As for exact data splits, each category is represented  between 8% and 11% per each of the splits.


The file "PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_Lambda3.ipynb"  was to test out Three Different Lambda values and different epoch-periods.

From what we knew about previous results using just different lambda values, it seemed like using a weight of 1, 100, and 1000 respectively required {~25} {~15*} {~45 epochs with early stopping} when given 45 epochs and a patience of 10 for early stopping (*30 epochs and patience of 3).  So we decided to split this period up into **Three Phases** Training Phase 1: 5 Epochs, Training Phase 2: 5 Epochs, Training Phase 3: 25 Epochs with Early Stopping. and Patience of 0.

 
**Table of  Misclassifications with Different Lambda-Combos**
The below graphic shows 27 unique combinations of the three-phases and three-lambda values 3^3, and the associated misclassifications.
<img src = "https://user-images.githubusercontent.com/13596380/231316234-047d6483-cfd2-4f11-bad6-34ed64688bdc.png" alt="Table of Values" width="50%" title="Table of Misclass Values for Different Lambda Values">


**Graph of  Misclassifications with Different Lambda-Combos **
<img src = 
"https://user-images.githubusercontent.com/13596380/231319406-9ae3be38-18df-4db9-bb32-813db7ed1d43.png" alt="Colored First Phase Image" width="70%"  title="Misclassifications with 27 Unique Lambda-Combos">

However at this point, it looks like only the initial-phase does anything to effect the Misclassifications 9t_4P.  But plotting the other second phase values does seem to show more variability in 4T_9P then was seen with just static lambda values:
![image](https://user-images.githubusercontent.com/13596380/231481503-e51cf4a7-403d-4b20-b3bc-dcd789a508b1.png)


Summary:
After running 27 unique combinations of three lambda values across different values, it was clear that only the initial phase (first five epochs) seemed to be the most important factor.  The second and third phase didn't seem to have as large of an effect, if any.  Dr Cs guidance at this point was to explore more lambda-values in the first-five-epochs.  

----
#### Realization there is probably a coding error

However, after testing more combinations, I start to wonder if I have a coding error, because, nothing seems to change significantly after any epoch, once the model architecture is trained with a misclassification value, nothing seems to change no matter what the value is changed to.   To see this I loaded the results of the previous 27 unique combos and graphed the loss values.   I would think that if the weight-of-a-misclassification had a large impact, the overall loss-value would be larger overall for the epoch that changed.     According to the data, it doesn't appear like that was happening.   At epoch 5 and Epoch 10  I thought I would see some sort of jump in the loss value but i do not.  The picture below is from this [notebook](https://colab.research.google.com/github/RachelRamirez/misclassification_matrix/blob/main/Load_and_Analyze_Results_of_pkl_files.ipynb#scrollTo=8My0DZuBOBG1&uniqifier=1) . So i go back and rewrite my code so that it has more of a 'fine-tuning' format on 4/18/2023 which is just to say I try a differenet way of loading the initial model and continue training it.
![image](https://user-images.githubusercontent.com/13596380/232867477-644dbb36-f82a-4337-81dd-8eb97c4147e2.png)


#### 
New Code that works, in [new workbook]( https://colab.research.google.com/github/RachelRamirez/misclassification_matrix/blob/main/PreExperiment_PA_Shfl_40D_Lambda1_Lambda2_FineTuning.ipynb#scrollTo=2kR-Az-v1r6b&uniqifier=2)

```
for i in range(10):

  cost_matrix[9,4] = 1
  model = create_model(cost_matrix)

  # I have to Recompile the created model with the New Cost Matrix (if it changes)
  model.compile(loss=WeightedCategoricalCrossentropy(cost_matrix), optimizer=rms,  metrics='categorical_accuracy',)
  
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights = True)


  X_train_shuffled = shuffle(X_train, random_state=42+i)  #PseudoRandom Seed based off for-loop
  Y_train_shuffled = shuffle(Y_train, random_state=42+i)


  model_history2 = model.fit(X_train_shuffled, Y_train_shuffled,          
                              batch_size=batch_size, epochs=nb_epoch, verbose=1,
                              validation_data=(X_val, Y_val), shuffle=True, use_multiprocessing=True, callbacks = [es_callback, cm_callback])

  cm3 = return_cm(model)    #returns Confusion Matrix

  model_history_all.append(model_history2)   #Add this history to the list
  cm_all.append(cm3)                         #Add this confusion matrix to the list

print(model_history_all)                     #Returns all for-loop HistoryObjects
```

----------
As of 5/1/2023 I've replaced RMS optimizer with SGD optimization.  The learning rate has to be pretty high 0.1, in order to execute within 100 epochs.  
