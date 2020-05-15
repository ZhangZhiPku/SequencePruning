from abc import abstractmethod
from src.models.Pruners import select_k, compress, LSTMSelfCompressionModule
import torch
import torch.nn


def generate_hard_mask(y_soft: torch.Tensor, remain_tokens_num):
    _, sorted_idx = y_soft.sort()
    _, sorted_idx = sorted_idx.sort()

    y_hard = sorted_idx < remain_tokens_num

    return y_hard


class LSTMClassificationModel(torch.nn.Module):

    def __init__(self,
                 hidden_size,
                 num_of_layer,
                 num_of_label,
                 num_of_tokens,
                 embedding_size,
                 embedding_matrix=None,
                 compression_module: torch.nn.Module = None):

        super().__init__()

        self.compression_module = compression_module
        self.embedding_layer = torch.nn.Embedding(
            embedding_dim=embedding_size,
            num_embeddings=num_of_tokens
        )
        if embedding_matrix is not None:
            self.embedding_layer.weight.data.copy_(torch.Tensor(embedding_matrix))

        self.lstm_modules = [
            torch.nn.LSTM(batch_first=True, input_size=embedding_size,
                          hidden_size=hidden_size, num_layers=1,
                          bidirectional=False)
            for _ in range(num_of_layer)
        ]
        for i, m in enumerate(self.lstm_modules):
            self.add_module('internal_lstm_%d' % i, m)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_of_label)
        )

    def get_embedding(self, input_sequence: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_sequence)

    def embedding_forward(self, embedding_sequence: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.lstm_module(embedding_sequence)
        return self.fc(hidden_states[:, -1, :])

    def compress(self, ts, es, pruning_rate):
        remaining_units = int(es.size()[1] * (1 - pruning_rate))
        # remain units num must be positive.
        remaining_units = max(remaining_units, 1)

        selection_mask, _ = self.compression_module.forward(
            ts, es, pruning_rate)

        es = es[selection_mask]
        es = es.reshape((-1, remaining_units, self.embedding_layer.embedding_dim))

        ts = ts[selection_mask]
        ts = ts.reshape((-1, remaining_units))

        return es, ts

    def sentence_forward(self, ts: torch.Tensor, pruning_rate=0.5):
        es = self.get_embedding(ts)

        if pruning_rate > 0:
            if self.compression_module is None:
                raise Exception('No Compression Module assigned to this model.')
            es, ts = self.compress(ts=ts, es=es, pruning_rate=pruning_rate)

        for lstm_module in self.lstm_modules:
            es, _ = lstm_module(es)

        pred_logits = self.fc(es[:, -1, :])
        return pred_logits

    def sentence_pair_forward(self, ta: torch.Tensor, tb: torch.Tensor, pruning_rate=0.5):
        ea = self.get_embedding(ta)
        eb = self.get_embedding(tb)

        if pruning_rate > 0:
            if self.compression_module is None:
                raise Exception('No Compression Module assigned to this model.')
            ea, ta = self.compress(ts=ta, es=ea, pruning_rate=pruning_rate)
            eb, tb = self.compress(ts=tb, es=eb, pruning_rate=pruning_rate)

        for lstm_module in self.lstm_modules:
            ea, _ = lstm_module(ea)
            eb, _ = lstm_module(eb)

        pred_logits = self.fc(ea[:, -1, :] - eb[:, -1, :])
        return pred_logits