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


# Load all training accounts into one dataframe
train_users_desc = pd.read_csv('training_data.csv', encoding='latin1')[['description', 'bot']]
train_users_desc.fillna('', inplace=True)
print('train total: {}'.format(train_users_desc.shape))

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
testdata['nb_guess'] = pd.read_csv('nb_test_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
testdata['svm_guess'] = pd.read_csv('svm_test_guess.csv', header=None)

# removing unnecessary columns. keeping only numbers for this part
testdata = testdata.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name','bot'], 1)

for c in testdata.columns:
    if(testdata[c].dtype==object):
        testdata[c] = testdata[c].replace(['TRUE','FALSE','None'], [1,0,np.NaN])

# testdata = testdata.dropna() #SHOULD WE?
# print(testdata)

for c in testdata.columns:
    if(testdata[c].dtype==object):
        testdata[c] = (testdata[c]).astype(np.float64)

for c in testdata.columns:
    testdata[c] = testdata[c].replace([np.NaN],testdata[c].mean(skipna=True, axis=0))
'''
'''

## XGBOOST
# load data
traindata = pd.read_csv('training_data.csv', sep=",", encoding='latin1')
traindata.fillna('', inplace=True)

# add a column 'nb_guess' with a Naive Bayes classification of the description
traindata['nb_guess'] = pd.read_csv('nb_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
traindata['svm_guess'] = pd.read_csv('svm_guess.csv', header=None)

traindata = traindata.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)

X_train = traindata.drop('bot',1)
y_train = traindata['bot']

# ROW SAMPLING
# grid search
model = XGBClassifier()
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(subsample=subsample)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X_train, y_train)

bScores = cross_val_score(grid_result.best_estimator_, X_train, y_train, cv=10)
print("cross val accuracy for row sampling:",np.mean(bScores))

testdata.columns = ['followers_count', 'friends_count', 'listedcount', 'favourites_count',
       'verified', 'statuses_count', 'nb_guess', 'svm_guess']

print("test:",testdata.columns)
print("train:",traindata.columns)

y_predicted = grid_result.predict(testdata)
print(y_predicted.dtype)

answer = pd.DataFrame(y_predicted.astype(int),testdata.index.values.astype(int)+1,['Bot'],dtype=int)
print(answer.dtypes,'\n')
answer.to_csv('results_file.csv',index=True,index_label='Id')

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# plot
# pyplot.figure()
# pyplot.errorbar(subsample, means, yerr=stds)
# pyplot.title("XGBoost subsample vs Log Loss")
# pyplot.xlabel('subsample')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('subsample.png')

# # COLUMN SAMPLING
# # grid search
# model = XGBClassifier()
# colsample_bytree = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# param_grid = dict(colsample_bytree=colsample_bytree)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_result = grid_search.fit(X, y)
#
# bScores = cross_val_score(grid_result.best_estimator_, X, y, cv=10)
# print("cross val accuracy for col sampling:",np.mean(bScores))
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
#
# # plot
# pyplot.figure()
# pyplot.errorbar(colsample_bytree, means, yerr=stds)
# pyplot.title("XGBoost colsample_bytree vs Log Loss")
# pyplot.xlabel('colsample_bytree')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('colsample_bytree.png')
#
#
# # SPLIT SAMPLE
# # grid search
# model = XGBClassifier()
# colsample_bylevel = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
# param_grid = dict(colsample_bylevel=colsample_bylevel)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_result = grid_search.fit(X, y)
#
# bScores = cross_val_score(grid_result.best_estimator_, X, y, cv=10)
# print("cross val accuracy for split sampling:",np.mean(bScores))
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
#
# # plot
# pyplot.figure()
# pyplot.errorbar(colsample_bylevel, means, yerr=stds)
# pyplot.title("XGBoost colsample_bylevel vs Log Loss")
# pyplot.xlabel('colsample_bylevel')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('colsample_bylevel.png')