import pandas as pd
import numpy as np
import pickle
from text_normalize import normalize
from textblob.classifiers import NaiveBayesClassifier

# Load all accounts into one dataframe
users = pd.read_csv('training_data.csv', encoding='latin1')
users.fillna('', inplace=True)

# #Split into training and test sets (80/20)
# split = np.random.rand(len(users)) < 0.8
# users_train = users[split]
# users_test = users[~split]

# # Train a Naive Bayes Classifier (NLTK implementation) on normalized descriptions
# desc_tuples = [tuple((normalize(row[5]), row[20])) for row in users_train.itertuples()]
# nbcl = NaiveBayesClassifier(desc_tuples)
#
# # Check the classifier's predictions for the test set
# nbcl_acc = nbcl.accuracy([tuple((normalize(row[5]), row[20])) for row in users_test.itertuples()])
# print('NB description classifier: {}'.format(nbcl_acc))
#
# # 'Pickle' the classifier for future use
# f = open('nbcl.pickle', 'wb')
# pickle.dump(nbcl, f)
# f.close()

# Load Pickled Classifier
loaded_nbcl = pickle.load(open('nbcl.pickle', 'rb'))

# Classify all descriptions and write class values to file
pd.DataFrame(loaded_nbcl.classify(normalize(desc)) for desc in users['description']).to_csv('nb_guess.csv', index=False,
                                                                                            header=False)
