import logging
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

from collections import defaultdict
from src.models.Lstm import *
from src.models.Pruners import *
from src.Metrics import compute_metrics
from src.Recorder import write_record

logger = logging.getLogger(__name__)


def display_compression_result(token_sequence, compression_mask, tokenizer):
    def reconstruct_sequence(token_sequence, tokenizer):
        untokenized_sequence = []
        for token in token_sequence:
            if token > 0: untokenized_sequence.append(tokenizer.index_word[token])
        return untokenized_sequence

    def mask_select_sequence(token_sequence, mask_sequence):
        sub_sequence = []
        for token, mask in zip(token_sequence, mask_sequence):
            if mask:
                sub_sequence.append(token)
            else:
                sub_sequence.append('[MASKED]')
        return sub_sequence

    unchanged_sequence = reconstruct_sequence(token_sequence, tokenizer)
    compressed_sequence = mask_select_sequence(unchanged_sequence, compression_mask)

    print('UNCOMPRESSED SENTENCE')
    for word in unchanged_sequence:
        print(word, end='\t')
    print('')
    print('COMPRESSED SENTENCE')
    for word in compressed_sequence:
        print(word, end='\t')
    print('')


def collate_fn(features, max_padding_length=128):
    text_ids, labels = [], []
    text_a, text_b = [], []

    for feature in features:
        text_a.append(feature['text_a'])
        if not feature['text_b']: text_b.append([0])
        else: text_b.append(feature['text_b'])
        labels.append(feature['label'])

    # dynamic padding
    dynamic_padding_length_a = max([len(_) for _ in text_a])
    dynamic_padding_length_a = min(dynamic_padding_length_a, max_padding_length)

    dynamic_padding_length_b = max([len(_) for _ in text_b])
    dynamic_padding_length_b = min(dynamic_padding_length_b, max_padding_length)

    text_a_ids = [[0] * (dynamic_padding_length_a - (len(_))) + _[:dynamic_padding_length_a] for _ in text_a]
    text_b_ids = [[0] * (dynamic_padding_length_b - (len(_))) + _[:dynamic_padding_length_b] for _ in text_b]

    return {
        'text_a': torch.tensor(text_a_ids).long(),
        'text_b': torch.tensor(text_b_ids).long(),
        'labels': torch.tensor(labels).long()
    }


def train(model, num_of_epoch, batchsize, dataset, recorder,
          pruning_rate_start=0.0, pruning_rate_end=0.0,
          device='cuda'):

    extract_loss_value = lambda x: x.detach().cpu().numpy()

    model = model.to(device=device)

    data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batchsize, shuffle=True)
    optimizer = Adam(params=model.parameters(), lr=1e-3)

    loss_fn = torch.nn.CrossEntropyLoss()

    num_of_step = num_of_epoch * len(data_loader)
    compression_rate = pruning_rate_start
    step = 0

    for epoch_i in tqdm(range(num_of_epoch), total=num_of_epoch, desc='EPOCH: '):
        for batch_data in tqdm(data_loader, total=len(data_loader), desc='BATCH: '):

            optimizer.zero_grad()

            token_sequence_a = batch_data['text_a'].long().to(device)
            token_sequence_b = batch_data['text_b'].long().to(device)

            label_sequence = batch_data['labels'].to(device)

            pred_logits = model.sentence_pair_forward(
                ta=token_sequence_a, tb=token_sequence_b, pruning_rate=compression_rate
            )

            loss = loss_fn(pred_logits, label_sequence)

            loss.backward()
            loss_value = extract_loss_value(loss)

            optimizer.step()

            step += 1
            if step % 25 == 0:
                print('runing at step %d, loss: %.4f, compression rate: %.2f' % (step, loss_value, compression_rate))
                compression_rate = (step / num_of_step) * (pruning_rate_end - pruning_rate_start)
                compression_rate += pruning_rate_start

            recorder['global_step'].append(len(recorder['global_step']) + 1)
            recorder['local_step'].append(step)
            recorder['loss'].append(loss_value)
            recorder['compression_rate'].append(compression_rate)


def eval(model: LSTMClassificationModel,
         batchsize: int, dataset: Dataset, task_name: str, recorder,
         pruning_rate: float = 0.0):

    data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batchsize)

    pred_labels = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, total=len(data_loader), desc='BATCH: '):

            token_sequence_a = batch_data['text_a'].cuda()
            token_sequence_b = batch_data['text_b'].cuda()

            pred_logits = model.sentence_pair_forward(
                ta=token_sequence_a, tb=token_sequence_b, pruning_rate=pruning_rate
            )

            pred_label = pred_logits.argmax(dim=1)
            pred_labels.append(pred_label.detach().cpu().numpy())

    pred_labels = np.concatenate(pred_labels, axis=0)
    real_labels = [_['label'] for _ in dataset]

    recorder['Metric'].append(compute_metrics(task_name, pred_labels, real_labels))
    print('Classify Accuracy: %s' % compute_metrics(task_name, pred_labels, real_labels))


def eval_with_seqlen(
        model: LSTMClassificationModel,
        batchsize: int, dataset: Dataset, task_name: str, recorder,
        pruning_rate: float = 0.0, bins: int = 5):
    data_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batchsize)

    pred_labels = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader, total=len(data_loader), desc='BATCH: '):

            token_sequence_a = batch_data['text_a'].cuda()
            token_sequence_b = batch_data['text_b'].cuda()

            pred_logits = model.sentence_pair_forward(
                ta=token_sequence_a, tb=token_sequence_b, pruning_rate=pruning_rate
            )

            pred_label = pred_logits.argmax(dim=1)
            pred_labels.append(pred_label.detach().cpu().numpy())

    pred_labels = np.concatenate(pred_labels, axis=0)
    real_labels = np.array([feature['label'] for feature in dataset])
    seq_lengths = np.array([len(feature['text_a']) + len(feature['text_b']) for feature in dataset])
    seq_argsort = np.argsort(seq_lengths)

    # split dataset into longer dataset and shorter dataset.
    bin_interval = len(pred_labels) // bins
    for bid in range(bins):
        bin_idx = seq_argsort[: bin_interval]
        bin_pl, bin_rl = pred_labels[bin_idx], real_labels[bin_idx]
        seq_argsort = seq_argsort[bin_interval: ]

        recorder['Bin %s Result' % bid].append(compute_metrics(task_name, bin_pl, bin_rl))
        print('Bin %s Result: %s' % (bid, compute_metrics(task_name, bin_pl, bin_rl)))


def main():
    task_names = ['sst-2', 'mrpc', 'qqp', 'mnli']
    pruning_methods = ['frequency', 'head', 'random', 'lstm']
    target_prune_rate = 0.9

    for task_name in task_names:
        for pruning_method in pruning_methods:
            record_file = 'records/lstm/record of %s(%s)' % (task_name, pruning_method)
            data_file = 'cache/lstm/%s' % task_name

            with open(data_file, 'rb') as file:
                saved = pickle.load(file)
                train_df        = saved['train_features']
                test_df         = saved['dev_features']
                tokenizer       = saved['tokenizer']
                num_of_labels   = saved['num_of_labels']

            model = LSTMClassificationModel(compression_module=None, hidden_size=300, num_of_layer=1, num_of_label=num_of_labels,
                                            embedding_size=300, num_of_tokens=min(tokenizer.num_words, (len(tokenizer.word_index)+1)))
            if pruning_method != 'lstm':
                pruner = PRUNERS[pruning_method]()
            else:
                pruner = LSTMSelfCompressionModule(lstm_module=model.lstm_modules[0])
            model.compression_module = pruner

            model = model.cuda()
            training_recorder = defaultdict(list)
            evaluation_recorder = defaultdict(list)

            train(
                model=model,
                num_of_epoch=3,
                batchsize=256 if task_name is not 'mrpc' else 64,
                dataset=train_df,
                recorder=training_recorder,
                pruning_rate_start=target_prune_rate,
                pruning_rate_end=target_prune_rate
            )

            eval(
                model=model,
                batchsize=256 if task_name is not 'mrpc' else 64,
                dataset=test_df,
                pruning_rate=target_prune_rate,
                recorder=evaluation_recorder,
                task_name=task_name
            )

            eval_with_seqlen(
                model=model,
                batchsize=256 if task_name is not 'mrpc' else 64,
                dataset=test_df,
                pruning_rate=target_prune_rate,
                recorder=evaluation_recorder,
                task_name=task_name
            )

            write_record(record_file, training_recorder, evaluation_recorder)


if __name__ == '__main__':
    main()