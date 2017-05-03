import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print('##### Data Info ##### ')

df = pd.read_csv('training_data.csv', sep=",", encoding='latin1')
df.fillna('', inplace=True)
print('total: {}'.format(df.shape))

# add a column 'nb_guess' with a Naive Bayes classification of the description
df['nb_guess'] = pd.read_csv('nb_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
df['svm_guess'] = pd.read_csv('svm_guess.csv', header=None)

'''
Other things to consider:
    lexical diversity,
    n-grams,
    friends/followers ratio,
    adaboost, <- implemented below
    preprocess svm descriptions
    use random forests only - without boosting
    use svm without nb
    look for 'bot' in name
'''

# removing unnecessary columns. keeping only numbers for this part
df = df.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)
print(df.columns)

X = df.drop('bot', 1)
Y = df['bot']

rfc = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(10, 100, 10),
              'max_features': ['sqrt', 'log2', None],
              'criterion': ['gini', 'entropy'],
              }

'''
All Parameters in a RFC
            'max_depth':[None],
            'min_samples_split':[2],
            'min_samples_leaf':[1],
            'min_weight_fraction_leaf':[0.0],
            'max_leaf_nodes':[None],
            'min_impurity_split':[1e-07],
            'bootstrap':[True],
            'oob_score':[False],
            'n_jobs':[1],
            'random_state':[None],
            'verbose':[0],
            'warm_start':[False],
            'class_weight':[None]
'''
# # Performing a Grid Search to find best RFC
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
# CV_rfc.fit(X, Y)
#
# print('Grid Searched RF')
# print(CV_rfc.best_params_)

from sklearn.model_selection import cross_val_score

## RUN after first calculating the best parameters from the code above.
## USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
best_rfc = RandomForestClassifier(max_features='sqrt', n_estimators=60, criterion='gini')
scores = cross_val_score(best_rfc, X, Y, cv=10)
print('crossvalidated accuracy: {}'.format(scores.mean()))
print()

from sklearn.ensemble import AdaBoostClassifier

brfc = AdaBoostClassifier(best_rfc,
                          algorithm="SAMME.R",
                          n_estimators=60)
bScores = cross_val_score(brfc, X, Y, cv=10)
print('crossvalidated accuracy after Adaboost: {}'.format(bScores.mean()))
print()
