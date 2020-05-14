import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_processors as processors
from src.Metrics import compute_metrics
from src.models.Pruners import PRUNERS
from src.Recorder import write_record


logger = logging.getLogger(__name__)
tensor2numpy_fn = lambda x: x.detach().cpu().numpy()


def train(BERT: BertForSequenceClassification, num_of_epoch, batchsize,
          task_name, dataset, recorder, pruning_method='random',
          pruning_schedule=None, pruners=None):

    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # speicial training optimizater for BERT
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in BERT.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in BERT.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * num_of_epoch
    )

    step = 0
    for epoch_i in tqdm(range(num_of_epoch), total=num_of_epoch, desc='EPOCH: '):
        for batch in tqdm(data_loader, total=len(data_loader), desc='BATCH: '):

            batch = tuple(t.cuda() for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3] if task_name != 'sts-b' else batch[3].float(),
                "pruners": pruners,
                "pruning_schedule": pruning_schedule,
                "pruning_method": pruning_method
            }

            outputs = BERT(**inputs)
            BERT.zero_grad()

            loss = outputs[0]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(BERT.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            step += 1
            loss_value = tensor2numpy_fn(loss)
            if step % 25 == 0:

                print('runing at step %d, loss: %.4f' %
                      (step, loss_value))

            recorder['global_step'].append(len(recorder['global_step']) + 1)
            recorder['local_step'].append(step)
            recorder['loss'].append(loss_value)


def evaluation(BERT: BertForSequenceClassification, batchsize,
               dataset, recorder, task_name, require_gradients=False,
               pruning_method='random', pruning_schedule=None, pruners=None):

    def _evaluation(BERT: BertForSequenceClassification):
        data_loader = DataLoader(dataset, batch_size=batchsize)
        BERT = BERT.cuda()

        pred_labels = []
        for batch in tqdm(data_loader, total=len(data_loader), desc='BATCH: '):

            batch = tuple(t.cuda() for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3] if task_name != 'sts-b' else batch[3].float(),
                "pruners": pruners,
                "pruning_schedule": pruning_schedule,
                "pruning_method": pruning_method
            }
            loss, pred_logits = BERT(**inputs)[: 2]

            if require_gradients:
                # must invoke backward function to release the graph.
                loss.backward()
                # gradient based pruning needs to flush gradients.
                BERT.zero_grad()

            pred_label = pred_logits.argmax(dim=1) if task_name != 'sts-b' else pred_logits.squeeze(-1)
            pred_labels.append(pred_label.detach().cpu().numpy())

        pred_labels = np.concatenate(pred_labels, axis=0)
        real_labels = torch.cat([_[3].unsqueeze(0) for _ in dataset]).detach().cpu().numpy()

        recorder['Metric'].append(compute_metrics(task_name, pred_labels, real_labels))
        print('Classify Accuracy: %s' % compute_metrics(task_name, pred_labels, real_labels))

    if require_gradients:
        _evaluation(BERT)
    else:
        with torch.no_grad():
            _evaluation(BERT)


def load_dataset(cache_file_path):
    features = torch.load(cache_file_path)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def load_model(model_path, task_name, num_labels, convert_to_GPU=True):
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained(
        model_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=None,
    )

    # must set this to be true!
    config.output_attentions = True

    BERT = BertForSequenceClassification.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
        cache_dir=None,
    )
    if convert_to_GPU:
        BERT = BERT.cuda()
    return BERT


def main():
    cache_file_path = 'cache/bert/'
    model_path = 'bert/'

    task_names = ['mrpc', 'cola', 'mnli', 'qnli', 'sts-b', 'sst-2', 'qqp', 'rte']
    pruning_methods = ['random', 'tail', 'frequency', 'ideal emission rate', 'ideal gradients']

    for task_name in task_names:
        for pruning_method in pruning_methods:
            # You have to modifies code here to test another pruning plan.
            pruning_schedule = [0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            pruners = [PRUNERS[pruning_method]().cuda() if pr > 0 else None
                       for pr in pruning_schedule]
            use_student_pruner = pruning_method in {'ideal gradients'}

            record_file = 'records/bert/record of %s(%s)' % (task_name, pruning_method)
            train_file = cache_file_path + task_name + '.train'
            eval_file = cache_file_path + task_name + '.eval'

            train_dataset = load_dataset(train_file)
            eval_dataset = load_dataset(eval_file)

            processor = processors[task_name]()
            label_list = processor.get_labels()
            num_labels = len(label_list)

            BERT = load_model(
                model_path,
                task_name,
                num_labels,
                convert_to_GPU=True
            )

            from collections import defaultdict
            training_recorder = defaultdict(list)
            evaluation_recorder = defaultdict(list)

            train(
                BERT=BERT,
                num_of_epoch=0,
                batchsize=32,
                dataset=train_dataset,
                recorder=training_recorder,
                task_name=task_name,
                pruning_method=pruning_method,
                pruning_schedule=pruning_schedule,
                pruners=pruners
            )
            evaluation(
                BERT=BERT,
                batchsize=32,
                dataset=eval_dataset,
                recorder=evaluation_recorder,
                task_name=task_name,
                pruning_method=pruning_method,
                pruning_schedule=pruning_schedule,
                pruners=pruners,
                require_gradients=(pruning_method == 'ideal gradients')
            )
            if use_student_pruner:
                evaluation(
                    BERT=BERT,
                    batchsize=32,
                    dataset=eval_dataset,
                    recorder=evaluation_recorder,
                    task_name=task_name,
                    pruning_method='student',
                    pruning_schedule=pruning_schedule,
                    pruners=pruners
                )

            # save record file.
            write_record(record_file, training_recorder, evaluation_recorder)


if __name__ == "__main__":
    main()