This current code EEG_CNN & EEG_BRFC finds the patterns in the EEG data and tries to predict the possibilities of having neurodegenerative disease in this case Alzheimer's or
Fronto Temporal Dementia (FTD) and Healthy case with the help of CNN and Balanced Random Forest Classifier which had theirs pros and cons wrt to computational cost 
and performance metrics. Both of the models are used for current, but can decided based on the requirements.



Log:

Figure_1.png

time stamp: 29.12.2024
epochs = 20
batch size = 50
accuracy = 82.0832

0-A, 1-FD, 2-C
As there is class imbalance in the datasets [36A's, 23FD's, 29C's] which might be the reasons for mis-predicted classes.

choosing ensemble learning over oversampling even it can increase computational costs 
Reasons:
1. chances the data may not accurately represent the underlying patterns in EEG data on synthetic samples.
2. chances of overfitting and needs to check the weights again after oversampling to reduce the overfitting

checking with BalancedRandomForestClassifier

Figure_2.png

precision    recall  f1-score   support

           0       0.78      0.74      0.76      3703
           1       0.67      0.71      0.69      2068
           2       0.77      0.78      0.78      3042

    accuracy                           0.75      8813
   macro avg       0.74      0.74      0.74      8813
weighted avg       0.75      0.75      0.75      8813

Accuracy: 0.7478


