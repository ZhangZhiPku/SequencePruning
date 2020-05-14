# Use this script to generate formatted GLUE benchmark data for BERT
# Make sure your data is stored at ../../data/
# Converted data will be stored at ../../cache/bert/ make sure that directory exists.

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import BertTokenizer

import torch
import logging

logger = logging.getLogger(__name__)


def load_and_cache_examples(raw_file, cache_file_path, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    eval_cah_file = cache_file_path + '.eval'
    train_cah_file = cache_file_path + '.train'
    logger.info('PROCESSING FILE %s' % raw_file)

    label_list = processor.get_labels()
    eval_examples = processor.get_dev_examples(raw_file)
    train_examples = processor.get_train_examples(raw_file)

    eval_features = convert_examples_to_features(
        eval_examples,
        tokenizer,
        label_list=label_list,
        max_length=128,
        output_mode=output_mode,
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    train_features = convert_examples_to_features(
        train_examples,
        tokenizer,
        label_list=label_list,
        max_length=128,
        output_mode=output_mode,
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    torch.save(eval_features, eval_cah_file)
    torch.save(train_features, train_cah_file)


TASKS = ['CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI']
data_dir = 'data/'
output_dir = 'cache/bert/'

# vocab file path of BERT tokenizer.
tokenizer_word_file_path = 'bert/vocab.txt'

for task in TASKS:
    load_and_cache_examples(
        raw_file=data_dir + task,
        cache_file_path=output_dir + task.lower(),
        task=task.lower(),
        tokenizer=BertTokenizer(vocab_file=tokenizer_word_file_path)
    )
logger.info('ALL TASKS FINISHED.')
