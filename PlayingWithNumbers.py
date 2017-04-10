import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

print("##### Data Info ##### ")

df_bots = pd.read_csv('bots_data.csv', sep=",", encoding='latin1', na_values='?')
df_nonbots = pd.read_csv('nonbots_data.csv', sep=",", encoding='latin1', na_values='?')
# print(df_bots.head(5))

df = pd.concat([df_bots, df_nonbots], ignore_index=True)
print('total: {}'.format(df.shape))

# removing unnecessary columns. keeping only numbers for this part
df = df.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)
print(df.columns)

shuffled_df = df.reindex(np.random.permutation(df.index))

# split the shuffled data into 80/20
train_split = 0.8
train_df = shuffled_df[:int(np.floor(shuffled_df.shape[0] * train_split))]
test_df = shuffled_df[int(np.floor(shuffled_df.shape[0] * train_split)):]

print("train: {}, test: {}".format(train_df.shape, test_df.shape))

## Random Forests
print("\n##### Random Forests #####")
X = train_df.drop('bot', 1)
Y = train_df['bot']
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(criterion="entropy")
clf_RF.fit(X, Y)
print('accuracy on training data: {}'.format(clf_RF.score(X, Y)))
print('accuracy on test data: {}'.format(clf_RF.score(test_df.drop('bot', 1), test_df['bot'])))
print()

## Decision Tree
# create the classifier
print("\n##### Decision Trees #####")
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier(criterion='entropy')
clf_dt = clf_dt.fit(train_df.drop('bot', 1), train_df['bot'])

# testing the classifier
predict_train_dt = clf_dt.predict(train_df.drop('bot', 1))
predict_test_dt = clf_dt.predict(test_df.drop('bot', 1))

# print accuracies
print('accuracy on training data: {}'
      .format(accuracy_score(train_df['bot'], predict_train_dt)))
print('accuracy on test data: {}'
      .format(accuracy_score(test_df['bot'], predict_test_dt)))
print()

## Linear Regression doesnt make sense because well, either bot or not.

## Logistic Regression
print("\n##### Logistic Regression #####")
from sklearn.linear_model import LogisticRegression

X = train_df.drop('bot', 1)
y = train_df['bot']

# Set regularization parameter
for i, C in enumerate((100, 1, 0.01)):
    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
    clf_l1_LR.fit(X, y)
    clf_l2_LR.fit(X, y)

    print("C=%.2f" % C)
    print("- Training Data:")
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, y))
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

    print("- Test Data:")
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(test_df.drop('bot', 1), test_df['bot']))
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(test_df.drop('bot', 1), test_df['bot']))

    print()

## Naive Bayes - Bernoulli and Multnomial
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

X = train_df.drop('bot', 1)
Y = train_df['bot']

clf_BNB = BernoulliNB()
clf_MNB = MultinomialNB()

clf_BNB.fit(X, Y)
clf_MNB = clf_MNB.fit(X, Y)

print("\n##### Naive Bayes #####")
print("Bernoulli NB score on training data: {}".format(clf_BNB.score(X, Y)))
print("Multinomial NB score on training data: {}".format(clf_MNB.score(X, Y)))
print("Bernoulli NB score on test data: {}".format(clf_BNB.score(test_df.drop('bot', 1), test_df['bot'])))
print("Multinomial NB score on test data: {}".format(clf_MNB.score(test_df.drop('bot', 1), test_df['bot'])))
print()
