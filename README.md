# Credit_Risk_Analysis
 Columbia Data Science Module 17

## Overview
Working from an anonymized dataset of LendingClub borrowers, I have used several different machine learning algorithms to gauge thecredit risk of borrowers. Credit risk datasets typically contain far more low risk borrowers than high risk borrowers, and this dataset is no exception. Because of this, the key distinction between the six different algorithms are the methods employed in compensating for the relative paucity of high risk borrowers. (After doing some initial data cleaning, there were 347 high risk borrowers and 68,470 non-high risk borrowers. I will compare model performance using naive oversampling, SMOTE oversampling, Clustered Centroids undersampling, SMOTEENN oversampling/undersampling (combined), balanced random forest, and EasyEnsemble Classifier. 

## Resources
Software/tools: Jupyter Notebook, NumPy, Pandas, SKLearn, IMBLearn

Data: LoanStats_2019Q1.csv (located in the Resources folder of this repository)

## Results

* **Naive Random Oversampling**
![naive_oversampling.PNG](Resources/naive_oversampling.PNG)


The naive random oversampling method resulted in a balanced accuracy score of 0.60, an average precision score of 0.99, and an average recall score of 0.71. 

* **SMOTE Oversampling**
![smote_oversampling.PNG](Resources/smote_oversampling.PNG)

The SMOTE oversampling method resulted in a balanced accuracy score of 0.62, an average precision score of 0.99, and an average recall score of 0.71. 

* **Clustered Centroids Undersampling**
![clustered_centroids_undersampling.PNG](Resources/clustered_centroids_undersampling.PNG)

The Clustered Centroids undersampling method resulted in a balanced accuracy score of 0.51, an average precision score of 0.99, and an average recall score of 0.47. 

* **Combination (Over and Under) Sampling with SMOTEENN**
![smoteenn_over_and_undersampling.PNG](Resources/smoteenn_over_and_undersampling.PNG)

The SMOTEENN combination sampling method resulted in a balanced accuracy score of 0.65, an average precision score of 0.99, and an average recall score of 0.56. 

* **Balanced Random Forest Classifier**
![balanced_random_forest.PNG.PNG](Resources/balanced_random_forest.PNG)

The balanced random forest classifier method resulted in a balanced accuracy score of 0.96, an average precision score of 1.0, and an average recall score of 0.93. 

## Summary

