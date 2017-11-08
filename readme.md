# Fraud Identification - Udacity Machine Learning project

## Overview
---
The goal of this project is to analyse the data set made available as a result of the Enron fraud case from the 2000's, and attempt to identify patterns in the data that can assist in the prediction of future fraud cases in similar industries. In order to help identify the patterns and validate results, I have used a number of machine learning techniques to attempt to automatically classify individuals into either "Person of Interest" or "Non Person of Interest" categories based on financial and communication (email) data.

The remainder of this document outlines how I went about this analysis and what the result of my findings were.


## Available Data
---
The data set originally made available contains a fair amount of information about the Enron conspirators as well as a number of other employees. Some summary statistics for the data set:

* 146 records are available in the data
* 18 of those entries are known persons of interest (12.3% of the total data)
* Each record has up to 20 features available (not including the POI flag)
 * However one of those features is "`email_address`" which cannot be used to predict POI given that it is a direct label for an individual, and provides no meaningful data about fraud intent

The small number of records and even smaller number of known POIs is likely to present challenges in model accuracy / precision / recall as it will be difficult to generalise a model based on so few data points. Refer to the validation and evaluation section below for details on how I approached this challenge.


### Outliers
In analysing the data, 3 outliers were identified that must be removed before any processing could be performed:

1. A user named "`TOTAL`" which is a copy/paste issue from an excel spreadsheet or similar is grossly biasing the financial data, as it is a summation of all other salaries, etc (not an actual person)
2. A user named "`THE TRAVEL AGENCY IN THE PARK`", which is clearly a business and not a person, is also embedded in the data set and should be removed for this exercise
3. A user named "`OCKHART EUGENE E`" is found in the document with no identifying features (`NaN` for all data points), as a result he can/should be removed to avoid biasing the non POI pool

In addition, please refer to the following table for some interesting information regarding sparsity of the remaining features:

Feature | Unique Values | Null Values
--- | --- | --- | ---:
salary | 94 | 49 | 
to_messages | 87 | 57 | 
deferral_payments | 39 | 105 | 
loan_advances | 4 | 140 | 
bonus | 41 | 62 | 
restricted_stock_deferred | 18 | 126 | 
deferred_income | 44 | 95 | 
total_stock_value | 124  | 18 | 
expenses | 94 | 49 | 
from_poi_to_this_person | 57 | 69 | 
exercised_stock_options | 101 | 42 | 
from_messages | 65 | 57 | 
other | 91 | 52 | 
from_this_person_to_poi | 41 | 77 | 
long_term_incentives | 52 | 78 | 
shared_receipt_with_poi | 84 | 57 | 
restricted_stock | 91 | 34 | 
director_fees | 17 | 127 | 

I also found in the data there were 2 sets of typos that existed for BELFER ROBERT and BHATNAGAR SANJAY which transposed data into the wrong columns. I manually corrected this error using data from the provided PDF to ensure that the information being used in the model is as accurate as possible.

## Feature Selection / Feature Engineering
---
To improve the effectiveness of my machine learning models, I needed to reduce the diversity of features that I made available. Instead of manually analysing every feature individually, I allowed the classifier search to select between PCA (primary component identification) across the full feature set, or SelectKBest - tuned for what would deliver a maximal f1 value (proxy for recall and precision). The highest performing classifier was a <classifier name> leveraging <function> for dimensionality reduction.

This would analyse each features contribution to the model and only select the 8-10 most impactful features - re; the 8-10 range refer to the tuning selection below for details on the final value selected in this range.

<!-- TODO, add score values in for each feature -->
Feature | Score | Selected?
--- | --- | ---:
salary  |  | YES
bonus  |  | YES 
deferred_income   |  | YES
exercised_stock_options | | YES
shared_receipt_with_poi | | YES
total_payments | | YES
total_stock_value | |  YES
to_poi_pct | | YES
shared_receipt_with_poi | | YES

Based purely on the classifier types, scaling was not relevant for my model (as the GridSearch had to choose from classifiers that do not benefit from scaled data). However as I used PCA to further reduce the dimensionality of my data, I had to normalise all of the remaining feature values to ensure that large values did not grossly outweigh smaller values (e.g. comparing salary data to number of emails received. To do so, I used a QuantileTransformer which would reduce the impact of outliers and strive for a roughly normal distribution.

In addition to the existing features, I identified and engineered 2 of my own feaures - converting the number of emails sent to/from a person of interest to a percentage of the total emails that person has sent or received. This value would logically seem to be of more interest than an absolute value of emails as it allows more ready comparison between actors regardless of how heavily the individual uses email.

I ran my optimised models (see below) both with and without my features to determine whether they would be selected by SelectKBest and what scale of impact they would have on the resulting model scores.

Type | Accuracy | Precision | Recall | F1 | Winner?
--- | ---: | ---: | ---: | ---: | :---:
With my features | tbc | tbc | tbc | tbc | tbc
Without my features | tbc | tbc | tbc | tbc | tbc

As a result of the data above, even though I included the features in my feature_list, they were dropped from my mdoel by SelectKBest in the final pipeline.
<!--- TODO train 2 classifier sets, one with the new features and one without - and compare the results-->


## Algorithm Selection
---
The algorithm was autmatically selected by GridSearchCV, and the results of all the competing algorithm's can be seen below. I setup the GridSearch to optimise for f1 performance of the model to attempt to maximise for a combination of accuracy and precision which were the stated outcomes for this project.

Algorithm | Run time | Accuracy | Precision | Recall | F1 | Winner
--- | :---: | ---: | ---: | ---: | ---: | :---:
GaussianNB | 5.6s | 0.84853 | 0.41807 | 0.34700 | 0.37923 | No
KNearestNeighbors | 208.1s | 0.84107 | 0.32609 | 0.18000 | 0.23196 | No
NearestCentroid | 298.4s | 0.77680 | 0.33014 | 0.65500 | 0.43901 | Yes
DecisionTree | 756.6s | 0.85327 | 0.44254 | 0.38700 | 0.41291 | No
RandomForest |  |  |  |  |  |  |
AdaBoost |  |  |  |  |  |  |


## Parameter Tuning
---
Tuning refers to changing parameters of the selected algorithm to adjust how the model works and interacts with the provided data. If you do it wrong, you can end up negatively impacting the performance of your model. I tuned using GridSearch and a parameter dictionary, to automatically adjust and find the best combination of features that maximised for f1 (a proxy for recall + precision scores that we were aiming for).

Different parameters were tuned in each of the classifiers noted above. The below outlines all the testing performed.

**Gaussian (76 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | None
Dimensionality Reducer | [PCA(), SelectKBest()] | PCA
Number features (dim reduction) | range(2, 21) | 8

**NearestNeighbors (2,736 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | MaxAbsScaler
Dimensionality Reducer | [PCA(), SelectKBest()] | PCA
Number features (dim reduction) | range(2, 21) | 2
Number of neighbors | [1, 3, 5, 7, 9, 10, 15, 20, 50] |  3
Weight Function | ['uniform','distance'] |  distance
Power | [1, 2] | 2

**NearestCentroid (5,016 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | MaxAbsScaler
Dimensionality Reducer | [PCA(), SelectKBest()] | SelectKBest
Number features (dim reduction) | range(2, 21) | 6
Distance Metric | ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] | manhattan
Shrink Threshold |[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] | 5

**DecisionTree (8, 208 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | None
Dimensionality Reducer | [PCA(), SelectKBest()] | SelectKBest
Number features (dim reduction) | range(2, 21) | 13
Split quality criteria | ['entropy', 'gini'] | entropy
Split strategy | ['random', 'best'] | best
Minimum split samples | range(2, 20, 2) | 16
Max tree features | ['sqrt','log2', None] | None (all) 

**RandomForest (8,208 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | 
Dimensionality Reducer | [PCA(), SelectKBest()] |
Number features (dim reduction) | range(2, 21) | 
Number estimators | [5, 10, 20, 50] | 
Split quality criteria | ['entropy', 'gini'] | 
Minimum split samples | range(2, 20, 2) | 
Max tree features | ['auto','sqrt','log2', None] | 

**AdaBoost (380 combinations)**

Parameter Tested | Tested Values | Optimal Value
--- | --- | ---
Scaler | [None, MaxAbsScaler()] | 
Dimensionality Reducer | [PCA(), SelectKBest()] |
Number features (dim reduction) | range(2, 21) | 
Number estimators | [25, 50, 75, 100, 500] | 


## Model Validation
---
Validation refers to how you confirm that your model is predictive of the outcome as desired - either through splitting your data into training/testing sets, or through a form of cross-validation. A classic validation mistake is not splitting your data and testing directly on the same data that you trained on - causing wildly inaccurate (overstated) results.

Given the very small amount of data that was available for training and testing (146 records), a classic single train/test split was not advisable as we could introduce bias into the training or testing data sets by pure luck given that only 12% of the entries were true-positives. Therefore using a cross-validation / k-fold technique like StratifiedShuffleSplit was important as it allowed the training data to be cycled many times in small batches to train the model more thoroughly, and allow the biases in the dat to be averaged out over time.

A stratified split ensures that each run of the shuffle will generate a mutually exclusive set of training participants, that consumes the entire set of training data - while simultaneously randomising the data sets to help remove biases in the underlying training data.

This model was used in both the final tester.py class as well as in my GridSearch validation attempt to find the best classifier and hyperparameters to solve the problem.


## Model Evaluation
---
The goal of this project was to lift recall and precision above a threshold (0.3), in real terms this meant that I wanted to incorrectly accuse individuals of fraud less than 70% of the time (also known as a false positive) as well as incorrectly missing true-fraud (e.g. not calling someone a fraudster who actually is) less than 70% of the time.

In the final model that was submitted for this project, I ended up with the following average performances from the best model:
* Recall: x.xxxx -> meaning the model missed out on identifying x% of fraudsters
* Precision: x.xxxx -> meaning the model incorrectly tagged x% of entries as fraudsters who were innocent