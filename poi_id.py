#!/usr/bin/python

import sys
import pickle
import json
import numpy as np
import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MaxAbsScaler

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.cross_validation import StratifiedShuffleSplit

import warnings


# suppress warnings
warnings.filterwarnings('ignore')


# There was one user (EUGENE) with no data
def find_users_with_no_data(scan_data):
    no_data_users = []
    for ppl, features in scan_data.items():
        non_NaN = False
        for feature, value in features.items():
            if feature != 'poi':
                if value != 'NaN':
                    non_NaN = True
                    break

        if not non_NaN:
            no_data_users.append(ppl)

    return no_data_users


# There are no features with NaN set as the value for every user
def find_features_with_no_data(scan_data, feature_list):
    no_data_features = []
    for feature in feature_list:
        non_NaN = False
        for ppl, features in scan_data.items():
            try:
                if features[feature] != 'NaN':
                    non_NaN = True
                    break
            except KeyError:
                pass

        if not non_NaN:
            no_data_features.append(feature)

    return no_data_features


# Simple function to print out all the data from the data_dict
def display_values(data):
    # fields_of_interest = ['poi', 'deferral_payments', 'restricted_stock_deferred', 'total_stock_value', 'restricted_stock']
    fields_of_interest = ['restricted_stock_deferred']
    # persons_of_interest = ['BELFER ROBERT','BHATNAGAR SANJAY']
    persons_of_interest = []
    if len(persons_of_interest) > 0:
        for person in persons_of_interest:
            print(person, json.dumps(data[person], indent=4))
    else:
        for person, values in data.items():
            poi = data[person]['poi']
            for key, value in values.items():
                if key in fields_of_interest and value != 'NaN':
                    print(person, poi, key, value)


# Function to correct the entries in the data dictionary that were incorrectly set originally
def correct_invalid_values(data_to_correct):
    # There are some data input errors for BELFER ROBERT and BHATNAGAR SANJAY
    # This function corrects those errors back to their original state as per the provided PDF
    empty_value = 'NaN'

    data_to_correct['BELFER ROBERT']['deferred_income'] = -102500
    data_to_correct['BELFER ROBERT']['deferral_payments'] = empty_value
    data_to_correct['BELFER ROBERT']['expenses'] = 3285
    data_to_correct['BELFER ROBERT']['director_fees'] = 102500
    data_to_correct['BELFER ROBERT']['total_payments'] = 3285
    data_to_correct['BELFER ROBERT']['exercised_stock_options'] = empty_value
    data_to_correct['BELFER ROBERT']['restricted_stock'] = 44093
    data_to_correct['BELFER ROBERT']['restricted_stock_deferred'] = -44093
    data_to_correct['BELFER ROBERT']['total_stock_value'] = empty_value

    data_to_correct['BHATNAGAR SANJAY']['other'] = empty_value
    data_to_correct['BHATNAGAR SANJAY']['expenses'] = 137864
    data_to_correct['BHATNAGAR SANJAY']['director_fees'] = empty_value
    data_to_correct['BHATNAGAR SANJAY']['total_payments'] = 137864
    data_to_correct['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
    data_to_correct['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
    data_to_correct['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
    data_to_correct['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

    return data_to_correct


# Simple function to count how many non-zero values there are for each feature
def count_zeros_and_non_zeros(column):
    zeros = 0
    non_zeros = 0
    for entry in column:
        if entry == 0:
            zeros += 1
        else:
            non_zeros += 1

    return zeros, non_zeros


# Create some new features that show the % of emails sent and received by each individual to/from POIs
# These numbers may be more predictive than just absolute numbers as all individuals send different amounts of email
def create_new_features(existing_data):
    for person, features in existing_data.items():
        existing_data[person]['from_poi_pct'] = create_pct(features['from_poi_to_this_person'], features['to_messages'])
        existing_data[person]['to_poi_pct'] = create_pct(features['from_this_person_to_poi'], features['from_messages'])

    return existing_data


# Function to turn a number into a percent of a total value; used for feature engineering
def create_pct(partial, total):
    # Make sure the data is not NaN (if so return 0)
    if partial == 'NaN':
        partial = 0
    if total == 'NaN':
        return 0

    # If neither value is NAN, then return a percentage as a float value
    return (float(partial) / float(total)) * 100.


# Simple function to break the features and labels out of the original data dictionary
def split_features_from_labels(raw_data, target_features):
    new_data = featureFormat(raw_data, target_features, sort_keys=True)
    return targetFeatureSplit(new_data)


# Use the feature and labels data to try all possible combinations of parameters
# Return the best estimator (if it is performing above the baseline, and better than previous entries
# Otherwise return the previous estimator and score
def find_best_estimator_for_grid(classifier, pipe, params, data_set, feature_list, previous_score, previous_estimator):
    print('')
    print('-------------------------------------')
    print('--------- {classifier}'.format(classifier=classifier))
    print('-------------------------------------')
    print('')

    start = time.time()
    print(start)
    print('')

    np.random.seed(0)

    labels, features = split_features_from_labels(data_set, feature_list)

    cv = StratifiedShuffleSplit(labels, random_state=42)

    grid = GridSearchCV(pipe, param_grid=params, cv=cv, scoring='f1', n_jobs=1, verbose=1)
    grid.fit(features, labels)

    best_estimator = grid.best_estimator_

    # modified tester returns the scores for comparison purposes
    precision, recall, f1 = test_classifier(grid.best_estimator_, data_set, feature_list)
    if grid.best_params_.get('reduce_dim__k'):
        print('{x} features selected by SelectKBest'.format(x=grid.best_params_['reduce_dim__k']))
        for idx, feature in enumerate(feature_list[1:]):
            print(
                feature,
                grid.best_params_['reduce_dim'].scores_[idx],
                grid.best_params_['reduce_dim'].get_support()[idx]
            )
    else:
        print('PCA used for dim reduction; no features reportable.')

    # return the previous score and estimator if this is not better
    new_score = previous_score
    new_estimator = previous_estimator

    if precision >= 0.3 and recall >= 0.3 and f1 > previous_score:
        new_score = f1
        new_estimator = best_estimator

    print('-------------------------------------')
    end = time.time()
    log_time_to_complete(classifier, start, end)

    # Wait 60 seconds between runs, if this helps with the cores issue?
    time.sleep(60)

    return new_estimator, new_score


# Simple logger to track how long each classifier is taking to analyze
def log_time_to_complete(classifier, start_time, end_time):
    print('Took {time} seconds, to find best {classifier}'.format(classifier=classifier, time=end_time-start_time))


# Pulling in all features, will select best features automatically in pipeline (either PCA or SelectKBest)
features_list = [
    'poi',
    'salary', 'to_messages', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
    'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options',
    'from_messages', 'other','from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi',
    'restricted_stock', 'director_fees', 'from_poi_pct', 'to_poi_pct'
]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

####
# Previous validation checks done to fix data
# outlier exists for "TOTAL" row of data
####

# print(sorted(data_dict))
# print(find_users_with_no_data(data_dict))
# print(find_features_with_no_data(data_dict, features_list[1:]))

####
# Remove the invalid outliers completely
####
outliers = ['TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)

####
# Fix the erroneous data for 2 entries
####
# display_values(data_dict)
data_dict = correct_invalid_values(data_dict)

# Task 3: Create new feature(s)
data_dict = create_new_features(data_dict)

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
labels, features = split_features_from_labels(my_dataset, features_list)
best_estimator = None
best_score = 0.

# Hyperparameter tuning tests
# General parameters (all classifiers)
n_features = range(2, 21)
scalers = [None, MaxAbsScaler()]

# KNearestNeighbor parameters
n_neighbours = [1, 3, 5, 7, 9, 10, 15, 20, 50]
n_neighbor_weights = ['uniform','distance']
n_neighbor_p = [1, 2]

# Centroid parameters
centroid_distances = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
centroid_shrink = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# DecisionTree parameters
tree_criterion = ['entropy', 'gini']
tree_splitter = ['random', 'best']
tree_min_splits = range(2, 20, 2)
tree_max_features = ['sqrt', 'log2', None]

# RandomForest parameters
forest_estimators = [5, 10, 20, 50]

# AdaBoost parameters
boost_estimators = [25, 50, 75, 100, 500]

####
# Now try each different type of classifier to see how they perform independently
# Try out Gaussian NB
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', GaussianNB())
])
param_grid = [
    {
        'scale': scalers,
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features
    },
    {
        'scale': scalers,
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features
    }
]

best_estimator, best_score = find_best_estimator_for_grid('GaussianNB', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)

####
# Now try KNearestNeighbors
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', KNeighborsClassifier())
])
param_grid = [
    {
        'scale': scalers,
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features,
        'clf__n_neighbors': n_neighbours,
        'clf__weights': n_neighbor_weights,
        'clf__p': n_neighbor_p
    },
    {
        'scale': scalers,
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features,
        'clf__n_neighbors': n_neighbours,
        'clf__weights': n_neighbor_weights,
        'clf__p': n_neighbor_p
    }
]

best_estimator, best_score = find_best_estimator_for_grid('KNearestNeighbors', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)

####
#  Now try NearestCentroid
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', NearestCentroid())
])
param_grid = [
    {
        'scale': scalers,
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features,
        'clf__metric': centroid_distances,
        'clf__shrink_threshold': centroid_shrink
    },
    {
        'scale': scalers,
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features,
        'clf__metric': centroid_distances,
        'clf__shrink_threshold': centroid_shrink
    }
]

best_estimator, best_score = find_best_estimator_for_grid('NearestCentroid', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)


####
#  Now try DecisionTree
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42))
])
param_grid = [
    {
        'scale': scalers,
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features,
        'clf__criterion': tree_criterion,
        'clf__max_features': tree_max_features,
        'clf__splitter': tree_splitter,
        'clf__min_samples_split': tree_min_splits
    },
    {
        'scale': scalers,
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features,
        'clf__criterion': tree_criterion,
        'clf__max_features': tree_max_features,
        'clf__splitter': tree_splitter,
        'clf__min_samples_split': tree_min_splits
    }
]

best_estimator, best_score = find_best_estimator_for_grid('DecisionTree', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)


####
#  Now try RandomForest
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
param_grid = [
    {
        'scale': scalers,
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features,
        'clf__n_estimators': forest_estimators,
        'clf__max_features': tree_max_features,
        'clf__min_samples_split': tree_min_splits
    },
    {
        'scale': scalers,
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features,
        'clf__n_estimators': forest_estimators,
        'clf__max_features': tree_max_features,
        'clf__min_samples_split': tree_min_splits
    }
]

best_estimator, best_score = find_best_estimator_for_grid('RandomForest', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)

####
#  Now try AdaBoost
####
pipe = Pipeline([
    ('scale', MaxAbsScaler()),
    ('reduce_dim', PCA(random_state=42)),
    ('clf', AdaBoostClassifier(random_state=42))
])
param_grid = [
    {
        'reduce_dim': [PCA(random_state=42)],
        'reduce_dim__n_components': n_features,
        'clf__n_estimators': boost_estimators
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': n_features,
        'clf__n_estimators': boost_estimators
    }
]

best_estimator, best_score = find_best_estimator_for_grid('AdaBoost', pipe, param_grid, my_dataset, features_list, best_score, best_estimator)

####
# Write out the details of the absolutely best estimator and the score received (f1 value)
####
print(best_score, best_estimator)

####
# Now that we know what classifier is most effective, persist it to disk for testing
####
dump_classifier_and_data(best_estimator, my_dataset, features_list)
