import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

print('##### Data Info ##### ')

df_bots = pd.read_csv('bots_data.csv', sep=",", encoding='latin1')
df_nonbots = pd.read_csv('nonbots_data.csv', sep=",", encoding='latin1')
# print(df_bots.head(5))

df = pd.concat([df_bots, df_nonbots], ignore_index=True)
df.fillna('?', inplace=True)
print('total: {}'.format(df.shape))

# add a column 'nb_guess' with a Naive Bayes classification of the description
df['nb_guess'] = pd.read_csv('nb_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
df['svm_guess'] = pd.read_csv('svm_guess.csv', header=None)

# removing unnecessary columns. keeping only numbers for this part
df = df.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)
print(df.columns)

# split the dataset into 80/20
split = np.random.rand(len(df)) < 0.8
train_df = df[split]
test_df = df[~split]

print('train: {}, test: {}'.format(train_df.shape, test_df.shape))

X = train_df.drop('bot', 1)
Y = train_df['bot']

rfc = RandomForestClassifier()
param_grid = {'n_estimators': np.arange(10, 1000, 200),
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
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# CV_rfc.fit(X, Y)

print('CROSSVALIDATED RF')
# print(CV_rfc.best_params_)
# print(CV_rfc.cv_results_)


## RUN after first calculating the best parameters from the code above.
## USE THE BEST ONES IN THE CLASSIFIER'S ARGUMENTS
best_rfc = RandomForestClassifier(max_features='sqrt', n_estimators=200, criterion='entropy')
best_rfc.fit(X, Y)
print('accuracy on training data: {}'.format(best_rfc.score(X, Y)))
print('accuracy on test data: {}'.format(best_rfc.score(test_df.drop('bot', 1), test_df['bot'])))
print()
