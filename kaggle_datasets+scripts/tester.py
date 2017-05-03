import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from text_normalize import normalize
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib
from sklearn.svm import SVC
matplotlib.use('Agg')
from matplotlib import pyplot

def runNameAnalysis(df):
    df['contains_bot'] = df['screen_name'].str.contains('bot',case=False)*1

def calculateRatios(df):
    df['ff_Ratio'] = (df['friends_count']+1)/(df['followers_count']+1)

def assumeVerification(df):
    df.ix[df['verified'].isnull() & df['ff_Ratio']<0.04,'verified'] = 1
    df.ix[df['verified'].isnull() & df['ff_Ratio']>0.04,'verified'] = 0

    print(df.head(5))

def isEnglish(df):
    df['english'] = 0
    df.ix[df['lang'].isin(['en','en-gb']),'english'] = 1

# Load all training descriptions into one dataframe for NB and SVM
# train_users_desc = pd.read_csv('training_data.csv', encoding='latin1')[['description', 'bot']]
# train_users_desc.fillna('', inplace=True)
# print('train total: {}'.format(train_users_desc.shape))


# Load and fix training data
print('##### Data Info ##### ')
testdata = pd.read_csv('test_data_4_students.csv', sep=",", encoding='utf-8')
testdata.fillna('', inplace=True)
print('test total: {}'.format(testdata.shape))
testdata = testdata[:575]

'''
'''## NAIVE BAYES ANALYSIS
# Load Pickled Classifier
# loaded_nbcl = pickle.load(open('nbcl.pickle', 'rb'))

# Classify all descriptions and write class values to file
# pd.DataFrame(loaded_nbcl.classify(normalize(desc)) for desc in testdata['description']).to_csv('nb_test_guess.csv', index=False,
#                                                                                             header=False)

'''
'''

## SVM ANALYSIS
# # calculate the BOW representation
# vectorizer = CountVectorizer(min_df=1)
# word_counts = vectorizer.fit_transform(train_users_desc['description'])
#
# TFIDF
# tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
#
# X_train = tf_transformer.transform(word_counts)
# # X_train = word_counts #UNCOMMENT FOR BOW (BAG OF WORDS)
# y_train = train_users_desc['bot']
#
# clf = sklearn.svm.LinearSVC()
# clf.fit(X_train, y_train)

# # Classify all descriptions and write class values to file
# test_word_count = vectorizer.transform(desc for desc in testdata['description'])
# testdata_processed = tf_transformer.transform(test_word_count)
# pd.DataFrame(
#     clf.predict(testdata_processed)).to_csv(
#     'svm_test_guess.csv', index=False, header=False)
# # X_test = test_word_count #UNCOMMENT FOR BOW (BAG OF WORDS)

'''
'''

# add a column 'nb_guess' with a Naive Bayes classification of the description
# testdata['nb_guess'] = pd.read_csv('nb_test_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
testdata['svm_guess'] = pd.read_csv('svm_test_guess.csv', header=None)

# add a column 'name_analysis'
runNameAnalysis(testdata)

# do a language analysis thingy
isEnglish(testdata)

ids = testdata['id'].astype(np.int64)

# removing unnecessary columns. keeping only numbers for this part
testdata = testdata.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'description', 'created_at', 'name','bot','lang'], 1)

for c in testdata.columns:
    if(testdata[c].dtype==object):
        testdata[c] = testdata[c].replace(['TRUE','FALSE','None'], [1,0,np.NaN])

for c in testdata.columns:
    if(testdata[c].dtype==object):
        testdata[c] = (testdata[c]).astype(np.float64)

for c in testdata.drop('verified',1).columns:
    testdata[c] = testdata[c].replace([np.NaN],testdata[c].mean(skipna=True, axis=0))

testdata.columns = ['followers_count', 'friends_count', 'listedcount', 'favourites_count',
       'verified', 'statuses_count','svm_guess','contains_bot','english']


# CALCULATE RATIOS HERE - AFTER TEST DATA HAS BEEN FIXED
calculateRatios(testdata)
assumeVerification(testdata)


'''
'''

## XGBOOST
# load data
traindata = pd.read_csv('training_data.csv', sep=",", encoding='latin1')
traindata.fillna('', inplace=True)

# add a column 'nb_guess' with a Naive Bayes classification of the description
# traindata['nb_guess'] = pd.read_csv('nb_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
traindata['svm_guess'] = pd.read_csv('svm_guess.csv', header=None)

runNameAnalysis(traindata)

traindata = traindata.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)

calculateRatios(traindata)
# print(np.mean(traindata[traindata['verified']==True]['ff_Ratio']))  #mean = 0.04

X_train = traindata.drop('bot',1)
y_train = traindata['bot']


#########
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)
CV_rfc.fit(X_train, y_train)
print('Grid Searched RF')
print(CV_rfc.best_params_)

print("RF: score of the best estimator over the leftout set:",CV_rfc.best_score_)
print("RF: score of the best estimator over the complete set:",CV_rfc.best_estimator_.score(X_train,y_train))
# scores = cross_val_score(CV_rfc.best_estimator_, X_train, y_train, cv=10)
# print('crossvalidated accuracy: {}'.format(scores.mean()))
print()




##############

# ROW SAMPLING
# grid search
model = XGBClassifier()
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(subsample=subsample)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)

# bScores = cross_val_score(grid_result.best_estimator_, X_train, y_train, cv=10)
# print("cross val accuracy for row sampling:",np.mean(bScores))


print("RSXG: score of the best estimator over the leftout set:",grid_result.best_score_)
print("RSXG: score of the best estimator over the complete set:",grid_result.best_estimator_.score(X_train,y_train))
print()
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# # plot
# pyplot.figure()
# pyplot.errorbar(subsample, means, yerr=stds)
# pyplot.title("XGBoost subsample vs Log Loss")
# pyplot.xlabel('subsample')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('subsample.png')

# COLUMN SAMPLING
# grid search
model = XGBClassifier()
colsample_bytree = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(colsample_bytree=colsample_bytree)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)

print("CSXG: score of the best estimator over the leftout set:",grid_result.best_score_)
print("CSXG: score of the best estimator over the complete set:",grid_result.best_estimator_.score(X_train,y_train))
print()
# bScores = cross_val_score(grid_result.best_estimator_, X_train, y_train, cv=10)
# print("cross val accuracy for col sampling:",np.mean(bScores))

# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
#
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# plot
# pyplot.figure()
# pyplot.errorbar(colsample_bytree, means, yerr=stds)
# pyplot.title("XGBoost colsample_bytree vs Log Loss")
# pyplot.xlabel('colsample_bytree')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('colsample_bytree.png')


# SPLIT SAMPLE
# grid search
model = XGBClassifier()
colsample_bylevel = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(colsample_bylevel=colsample_bylevel)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)

# bScores = cross_val_score(grid_result.best_estimator_, X_train, y_train, cv=10)
# print("cross val accuracy for split sampling:",np.mean(bScores))
print("SSXG: score of the best estimator over the leftout set:",grid_result.best_score_)
print("SSXG: score of the best estimator over the complete set:",grid_result.best_estimator_.score(X_train,y_train))
print()

###### PREDICT AND CREATE RESULTS FILE
y_predicted = grid_result.predict(testdata)
answer = pd.DataFrame(y_predicted.astype(np.int64),ids.astype(np.int64),['Bot'])
answer.to_csv('results_file.csv',index=True,index_label='Id')
######


# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))

# plot
# pyplot.figure()
# pyplot.errorbar(colsample_bylevel, means, yerr=stds)
# pyplot.title("XGBoost colsample_bylevel vs Log Loss")
# pyplot.xlabel('colsample_bylevel')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('colsample_bylevel.png')