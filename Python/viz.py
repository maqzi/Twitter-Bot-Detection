import numpy as np
import pandas as pd

df_bots = pd.read_csv('bots_data.csv', sep=",", encoding='latin1')
df_nonbots = pd.read_csv('nonbots_data.csv', sep=",", encoding='latin1')
# print(df_bots.head(5))

df = pd.concat([df_bots, df_nonbots], ignore_index=True)
df.fillna('?', inplace=True)
print('total: {}'.format(df.shape))

# add a column 'nb_guess' with a Naive Bayes classification of the description
df['nb_guess'] = pd.read_csv('nb_guess.csv', header=None)

# removing unnecessary columns. keeping only numbers for this part
df = df.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)
print(df.columns)
# print(df.head(5))

import pylab as plt

# plt_df = df[['followers_count','bot']].plot(kind='scatter',x='followers_count',y='bot')
# plt.title('Followers vs Bots')
# plt.show()
# plt_df = df[['friends_count','bot']].plot(kind='scatter',x='friends_count',y='bot')
# plt.title('Friends vs Bots')
# plt.show()

# verified bots vs verified humans
verified_df = df[df['verified'] == 1]
print('verified bots: {}'.format(len(verified_df[df['bot'] == 1])))
print('verified humans: {}'.format(len(verified_df[df['bot'] == 0])))
