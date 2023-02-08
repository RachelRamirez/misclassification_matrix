# Readme

#### The first effort is [Reproducibility file](https://github.com/RachelRamirez/misclassification_matrix/blob/main/Reproducible_Misclassification_Cost_Matrix_Example.ipynb) was to show reproducible results using three identical neural networks using three different cost matrix functions techniques: the normal default one where there are no additional cost-matrix supplied, and then two Keras' users custom code on a Github Issue board using a cost matrix of all ones.  Once seeded the results between all three were identical, you just have to remember to compare them after restarting and running through once or comparing the correct order.

Once I knew the three methods were equivalent I started to experiment with weights. 

#### The second effort is [w_array[7, 9] = 1.5](https://github.com/RachelRamirez/misclassification_matrix/blob/main/%5B7%2C9%5D_Misclassification_Cost_Matrix_Example.ipynb) where I made  the highest misclassification rate according to the confusion matrix on the validation set, have a weighted value of 1.5.  This is where I am now, assessing results of 30 similarly seeded runs,  which show the variability in the training/validation process, possibly due to the batch order.  I want to make a histogram matrix plot to identify what the distributions look like in each of the confusion matrices. 
