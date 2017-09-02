import pandas as pd
import numpy as np
import pickle
from text_normalize import normalize
from textblob.classifiers import NaiveBayesClassifier

# Load all accounts into one dataframe
bots = pd.read_csv('bots_data.csv', encoding='latin1')
nonbots = pd.read_csv('nonbots_data.csv', encoding='latin1')
users = pd.concat([bots, nonbots], ignore_index=True)
users.fillna('', inplace=True)

# Split into training and test sets (80/20)
split = np.random.rand(len(users)) < 0.8
users_train = users[split]
users_test = users[~split]

# Train a Naive Bayes Classifier (NLTK implementation) on normalized descriptions
desc_tuples = [tuple((normalize(row[5]), row[20])) for row in users_train.itertuples()]
nbcl = NaiveBayesClassifier(desc_tuples)

# Check the classifier's predictions for the test set
nbcl_acc = nbcl.accuracy([tuple((normalize(row[5]), row[20])) for row in users_test.itertuples()])
print('NB description classifier: {}'.format(nbcl_acc))

# 'Pickle' the classifier for future use
f = open('nbcl.pickle', 'wb')
pickle.dump(nbcl, f)
f.close()

# Classify all descriptions and write class values to file
[nbcl.classify(normalize(desc)) for desc in users['description']].to_csv('nb_guess.csv', index=False)
