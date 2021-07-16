# Credit_Risk_Analysis
 Columbia Data Science Module 17

## Overview
Working from an anonymized dataset of LendingClub borrowers, I have used several different machine learning algorithms to gauge thecredit risk of borrowers. Credit risk datasets typically contain far more low risk borrowers than high risk borrowers, and this dataset is no exception. Because of this, the key distinction between the six different algorithms are the methods employed in compensating for the relative paucity of high risk borrowers. (After doing some initial data cleaning, there were 347 high risk borrowers and 68,470 non-high risk borrowers. I will compare model performance using naive oversampling, SMOTE oversampling, Clustered Centroids undersampling, SMOTEENN oversampling/undersampling (combined), balanced random forest, and EasyEnsemble Classifier. 

## Resources
Software/tools: Jupyter Notebook, NumPy, Pandas, SKLearn, IMBLearn

[Data:](https://github.com/perryabdulkadir/Credit_Risk_Analysis/blob/main/Resources/LoanStats_2019Q1.csv) LoanStats_2019Q1.csv 

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
![balanced_random_forest.PNG](Resources/balanced_random_forest.PNG)

The balanced random forest classifier method resulted in a balanced accuracy score of 0.80, an average precision score of 0.99, and an average recall score of 0.92. 

* **Easy Ensemble AdaBoost Classifier**
![easy_ensemble_classifier.PNG](Resources/easy_ensemble_classifier.PNG)

The Easy Ensemble AdaBoost Classifier method resulted in a balanced accuracy score of 0.91, an average precision score of 0.99, and an average recall score of 0.94. 


## Summary and Recommendation

To begin with, we should clarify which metric is most important between precision and recall. For this use case, recall (sensitivity) is more important than precision. It is a better scenario for the algorithm to flag a good applicant as high risk, only to have a manual review approve them (false positive) than it is to have the algorithm approve a high risk applicant (false negative). 

I recommend two models, each of which may be useful depending on the use case. The Easy Ensemble AdaBoost Classifier had the highest balanced accuracy score (0.91), had a precision score of 0.99, and had a recall score of 0.91. Therefore, this is the model to use if overall model performance is the greatest concern. The Balanced Random Forest Classifier performed a bit worse, with an accuracy score of 0.80, a precision score of 0.99, and an average recall score of 0.92. The benefit of this model, however, is that it is more understandable for humans as it is easy to see which features were most important to the model performance, using the code below.

```
imp = dict(zip(*[X.columns.tolist(), random_forest.feature_importances_]))
{k: v for k, v in sorted(imp.items(), key=lambda item: item[1], reverse=True)}
```

Additionally, while the balanced accuracy score is lower than the Easy Ensemble AdaBoost Classifier, its sensitivity is actually just slightly higher. If this is the most important feature for our analysis, then the random forest method has a slight edge. 

Both of these models could be valuable in the right situation. Initially, I suspected that the EasyEnsemble model may have been overfitted; that is, it reflects the idiosyncracies of this dataset but would not perform well if applied to new data. I checked for overfitting by seeing if model performance was significantly different on the training dataset that the model had seen before. On the training dataset, the balanced accuracy score was slightly higher at 0.96, the precision score was slightly higher, at 1.0, and the recall score was slightly higher at 0.94. The difference in performance is not so great that I believe overfitting is happening, but it would still be worthwhile to try this model on more data, if available, to test its performance.
