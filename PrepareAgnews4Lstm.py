# Use this script to generate formatted AGNews dataset for LSTM
# Make sure your data is stored at ../../data/ag_news_csv/
# Converted data will be stored at ../../cache/lstm/ make sure that directory exists.

import pandas as pd
import pickle as pkl
from tensorflow.python.keras.preprocessing.text import Tokenizer

data_dir = 'data/ag_news_csv/'
output_dir = 'cache/lstm/'

train_df = pd.read_csv(data_dir + 'train.csv', header=None,
                       names=['label', 'title', 'content'])
test_df = pd.read_csv(data_dir + 'test.csv', header=None,
                      names=['label', 'title', 'content'])

tokenizer = Tokenizer(num_words=65000, lower=True)
tokenizer.fit_on_texts(train_df['content'])

train_df['tokens'] = tokenizer.texts_to_sequences(train_df['content'])
test_df['tokens'] = tokenizer.texts_to_sequences(test_df['content'])

train_df = train_df.drop(['title', 'content'], axis=1)
test_df = test_df.drop(['title', 'content'], axis=1)

# Make label start with 0
# The original labels are start with 1
train_df['label'] = train_df['label'] - 1
test_df['label'] = test_df['label'] - 1

num_of_labels = len(set(train_df['label']))

with open(output_dir + 'agnews', 'wb') as file:
    pkl.dump([train_df, test_df, num_of_labels, tokenizer], file)
