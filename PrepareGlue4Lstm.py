# Use this script to generate formatted GLUE benchmark data for LSTM
# Make sure your data is stored at ../../data/
# Converted data will be stored at ../../cache/lstm/ make sure that directory exists.

import os
import logging
import pickle
from tqdm import tqdm

# Used for load data from textual file.
from transformers import glue_processors as processors
from transformers import InputExample

# Only need tokenizer from tensorflow.python.keras
# You can replace it with another tokenizer as you wish.
from tensorflow.python.keras.preprocessing.text import Tokenizer


def convert(example: InputExample) -> dict:
    return {
        'text_a': tokenizer.texts_to_sequences([example.text_a])[0],
        'text_b': tokenizer.texts_to_sequences([example.text_b])[0],
        'label': labels.index(example.label)  # convert string label to int.
    }


logger = logging.getLogger(__name__)
# Notice that task names are case sensitive
TASKS = ['MNLI', 'MRPC', 'QQP', 'SST-2']
for task in TASKS:
    logger.info('PROCESSING WITH %s:' % task)

    data_dir = 'data/' + task
    data_processor = processors[task.lower()]()

    # Load data from data_dir
    # Loaded data would be a list of transformers.InputExample
    dev_examples = data_processor.get_dev_examples(data_dir)
    train_examples = data_processor.get_train_examples(data_dir)

    # a list of labels, notice label here is a string.
    labels = data_processor.get_labels()

    # collect all text from training examples
    texts = []
    for example in train_examples:
        # example.text_b would be None somehow, so we add a cast here.
        texts.append(str(example.text_a))
        texts.append(str(example.text_b))

    # build tokenizer, the setting is fixed and enough for tasks in GLUE
    tokenizer = Tokenizer(num_words=65000, lower=True)
    tokenizer.fit_on_texts(texts)

    # start to translate tokens sequence to id sequence
    train_features = []
    for example in tqdm(train_examples, total=len(train_examples),
                        desc='PROCESSING WITH TRAIN EXAMPLES.'):

        if example.text_a is None:
            example.text_a = ''
        if example.text_b is None:
            example.text_b = ''

        train_features.append(convert(example))

    dev_features = []
    for example in tqdm(dev_examples, total=len(dev_examples),
                        desc='PROCESSING WITH DEVELOP EXAMPLES.'):

        if example.text_a is None:
            example.text_a = ''
        if example.text_b is None:
            example.text_b = ''

        dev_features.append(convert(example))

    # store processed data
    with open('cache/lstm/' + task.lower(), 'wb') as file:
        pickle.dump(
            {
                'train_features': train_features,
                'dev_features': dev_features,
                'tokenizer': tokenizer,
                'num_of_labels': len(labels)
            },
            file=file
        )
    logger.info('PROCESSED DATA STORE AT %s' % ('../../cache/lstm/' + task.lower()))

logger.info('ALL WORK FINISHED.')
