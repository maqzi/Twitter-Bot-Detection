import pandas as pd
from sklearn.svm import SVC
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pickle

# Load all accounts into one dataframe
bots = pd.read_csv('bots_data.csv', encoding='latin1')
nonbots = pd.read_csv('nonbots_data.csv', encoding='latin1')
users = pd.concat([bots, nonbots], ignore_index=True)[['description', 'bot']]
users.fillna('', inplace=True)

# Split into training and test sets (80/20)
split = np.random.rand(len(users)) < 0.8
users_train = users[split]
users_test = users[~split]

# calculate the BOW representation
vectorizer = CountVectorizer(min_df=1)
word_counts = vectorizer.fit_transform(users_train['description'])

# TFIDF
tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
X_train = tf_transformer.transform(word_counts)
# X_train = word_counts #UNCOMMENT FOR BOW (BAG OF WORDS)
y_train = users_train['bot']

# create classifier
clf = sklearn.svm.LinearSVC()
clf.fit(X_train, y_train)

# working on the test set
test_word_count = vectorizer.transform(users_test['description'])
X_test = tf_transformer.transform(test_word_count)
# X_test = test_word_count #UNCOMMENT FOR BOW (BAG OF WORDS)
y_predicted = clf.predict(X_test)
y_test = users_test['bot']

print("accuracy for SVM + TFIDF: {}".format(metrics.accuracy_score(y_test, y_predicted)))
# print("accuracy for SVM + BOW: {}".format(metrics.accuracy_score(y_test, y_predicted))) #UNCOMMENT FOR BOW (BAG OF WORDS)

# 'Pickle' the classifier for future use
f = open('svm.pickle', 'wb')
pickle.dump(clf, f)
f.close()

# Classify all descriptions and write class values to file
print(users.shape)
pd.DataFrame(clf.predict(tf_transformer.transform(vectorizer.transform(desc for desc in users['description'])))).to_csv(
    'svm_guess.csv', index=False, header=False)
