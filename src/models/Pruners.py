import torch


def compress(forward_sequence, soft_mask, compression_rate):
    ol, bs = forward_sequence.size()[1], forward_sequence.size()[0]
    cl = int(ol * compression_rate)

    hard_mask = select_k(soft_mask, k=ol - cl)

    return forward_sequence[hard_mask].reshape((bs, ol - cl, -1))


def select_k(y_soft: torch.Tensor, k, keep_first_unit=True):
    # if you are using BERT, be sure the first unit is kept.
    # smaller value will be kept with this function.
    if keep_first_unit:
        y_soft = y_soft.clone()
        y_soft[:, 0] = y_soft.min() - 1

    _, sorted_idx = y_soft.sort()
    _, sorted_idx = sorted_idx.sort()

    y_hard = sorted_idx < k

    return y_hard


class BERTStudentPruner(torch.nn.Module):

    def __init__(self, input_units):
        super(BERTStudentPruner, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_units, out_features=input_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_units, out_features=input_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_units, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs, compression_rate):
        remain_tokens_num = int(inputs.size()[1] * (1 - compression_rate))

        # remain tokens num must be positive.
        remain_tokens_num = max(remain_tokens_num, 1)

        y_soft = self.fc(inputs).squeeze(-1)
        y_hard = select_k(y_soft, remain_tokens_num)
        return y_hard, y_soft


class LSTMStudentPruner(torch.nn.Module):

    def __init__(self, input_units):
        super(LSTMStudentPruner, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_units, out_features=input_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_units, out_features=input_units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=input_units, out_features=1)
        )

    def forward(self, inputs, compression_rate):
        remain_tokens_num = int(inputs.size()[1] * (1 - compression_rate))

        # remain tokens num must be positive.
        remain_tokens_num = max(remain_tokens_num, 1)

        y_soft = self.fc(inputs).squeeze(-1)
        y_hard = select_k(y_soft, remain_tokens_num)
        return y_hard, y_soft


class TeacherPruner(torch.nn.Module):

    def __init__(self, use_teacher_output=True):
        super(TeacherPruner, self).__init__()

        self.student = None
        self.student_optimizer = None
        self.student_loss = None
        self.use_teacher_output = use_teacher_output

    def register_student_model(self, student):
        self.student = student
        self.student_optimizer = \
            torch.optim.Adam(params=self.student.parameters(), lr=1e-3)
        self.student_loss = torch.nn.MSELoss()

    def train_student(self, inputs, expected_out, normalize_dim=1,
                      divide_std=True, clear_grads=True):
        # Be aware that this function will clear student gradients automatically
        eps = 1e-7

        if self.student is None:
            raise Exception('No available student model.')
        _, pred_out = self.student.forward(inputs=inputs, compression_rate=0)

        # normalize output
        pred_out = (pred_out - pred_out.mean(dim=normalize_dim, keepdim=True))
        expected_out = (expected_out - expected_out.mean(dim=normalize_dim, keepdim=True))

        if divide_std:
            pred_out = pred_out / (pred_out.std(dim=normalize_dim, keepdim=True) + eps)
            expected_out = expected_out / (expected_out.std(dim=normalize_dim, keepdim=True) + eps)

        loss = self.student_loss(pred_out, expected_out)
        loss.backward(retain_graph=True)
        self.student_optimizer.step()
        if clear_grads:
            self.student.zero_grad()

    def student_forward(self, inputs, compression_rate):
        return self.student(
            inputs=inputs, compression_rate=compression_rate
        )


class LSTMSelfCompressionModule(TeacherPruner):

    def __init__(self, lstm_module: torch.nn.LSTM):
        super(LSTMSelfCompressionModule, self).__init__()
        self.internal_lstm_module = lstm_module

        self.input_size = lstm_module.input_size
        self.hidden_size = lstm_module.hidden_size

        self.register_student_model(LSTMStudentPruner(300))

    def cal_soft_logits(self, token_sequence, embedding_sequence):
        padding_mask = (token_sequence == 0)

        hs, (fh, _) = self.internal_lstm_module(embedding_sequence)

        # concat hidden states
        hs = torch.cat(
            [torch.zeros(size=[hs.size()[0], 1, hs.size()[-1]], dtype=torch.float, device=hs.device), hs],
            dim=1
        )
        hs = hs[:, :-1, :]

        W_xi, W_xf, W_xg, W_xo = self.internal_lstm_module.weight_ih_l0.view((4, self.hidden_size, self.input_size))
        W_hi, W_hf, X_hg, W_ho = self.internal_lstm_module.weight_hh_l0.view((4, self.hidden_size, self.hidden_size))
        b_xi, b_xf, b_xg, b_xo = self.internal_lstm_module.bias_ih_l0.view((4, self.hidden_size))
        b_hi, b_hf, b_hg, b_ho = self.internal_lstm_module.bias_hh_l0.view((4, self.hidden_size))

        # input gate outputs
        io = torch.sigmoid(
            torch.matmul(W_xi, embedding_sequence.reshape(-1, self.input_size).permute((1, 0))) +
            torch.matmul(W_hi, hs.reshape(-1, self.hidden_size).permute((1, 0))) +
            b_xi.unsqueeze(-1) + b_hi.unsqueeze(-1))
        # forget gate outputs
        fo = torch.sigmoid(
            torch.matmul(W_xf, embedding_sequence.reshape(-1, self.input_size).permute((1, 0))) +
            torch.matmul(W_hf, hs.reshape(-1, self.hidden_size).permute((1, 0))) +
            b_hf.unsqueeze(-1) + b_xf.unsqueeze(-1))
        # output gate outputs
        oo = torch.sigmoid(
            torch.matmul(W_xo, embedding_sequence.reshape(-1, self.input_size).permute((1, 0))) +
            torch.matmul(W_ho, hs.reshape(-1, self.hidden_size).permute((1, 0))) +
            b_ho.unsqueeze(-1) + b_xo.unsqueeze(-1))

        io, fo, oo = io.permute((1, 0)), fo.permute((1, 0)), oo.permute((1, 0))
        io, fo, oo = io.view_as(embedding_sequence), fo.view_as(embedding_sequence), oo.view_as(embedding_sequence)
        io, fo, oo = io.mean(dim=-1), fo.mean(dim=-1), oo.mean(dim=-1)

        compression_logits = 1 - io
        compression_logits += (padding_mask.float() if padding_mask is not None else 0)

        return compression_logits, hs

    def forward(self, ts, es, compression_rate=0.5):
        if self.use_teacher_output:
            soft_mask, hs = self.cal_soft_logits(ts, es)

            remain_tokens_num = int(ts.size()[1] * (1 - compression_rate))

            # remain tokens num must be positive.
            remain_tokens_num = max(remain_tokens_num, 1)

            y_hard = select_k(soft_mask, remain_tokens_num)

            # be careful with following statement.
            # do not train student model when inferencing.
            # otherwise it will cause an unexpected error.
            if self.training:
                self.train_student(inputs=es, expected_out=soft_mask, divide_std=False)
                # self.zero_grad()

            return y_hard, soft_mask
        else:
            return self.student_forward(es, compression_rate)


class TailCompressionModule(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_sequence, embedding_sequence, compression_rate):
        token_mask = token_sequence > 0

        remain_tokens_num = int(embedding_sequence.size()[1] * (1 - compression_rate))
        remain_tokens_num = max(remain_tokens_num, 1)

        position_idx = torch.range(start=1, end=embedding_sequence.size()[1],
                                   device=embedding_sequence.device, dtype=torch.long).unsqueeze(0)
        position_idx = position_idx - embedding_sequence.size()[1]
        position_idx = position_idx.repeat((embedding_sequence.size()[0], 1))

        position_idx *= token_mask

        y_hard = select_k(position_idx, remain_tokens_num)

        return y_hard, y_hard


class HeadCompressionModule(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_sequence, embedding_sequence, compression_rate):
        token_mask = token_sequence > 0
        remain_tokens_num = int(embedding_sequence.size()[1] * (1 - compression_rate))
        remain_tokens_num = max(remain_tokens_num, 1)

        position_idx = - torch.range(start=1, end=embedding_sequence.size()[1],
                                   device=embedding_sequence.device, dtype=torch.long).unsqueeze(0)
        position_idx = position_idx.repeat((embedding_sequence.size()[0], 1))
        position_idx *= token_mask

        y_hard = select_k(position_idx, remain_tokens_num)

        return y_hard, y_hard


class RandomCompressionModule(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_sequence, embedding_sequence, compression_rate):
        token_mask = token_sequence > 0
        remain_tokens_num = int(embedding_sequence.size()[1] * (1 - compression_rate))
        remain_tokens_num = max(remain_tokens_num, 1)

        y_soft = torch.rand(size=embedding_sequence.size()[: 2]).to('cuda')
        y_soft_with_mask = y_soft - token_mask.float() * 1
        y_hard = select_k(y_soft_with_mask, remain_tokens_num)

        return y_hard, y_soft


class FrequencyCompressionModule(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, token_sequence, embedding_sequence, compression_rate):

        remain_tokens_num = int(token_sequence.size()[1] * (1 - compression_rate))

        # remain tokens num must be positive.
        remain_tokens_num = max(remain_tokens_num, 1)

        y_hard = select_k(-token_sequence, remain_tokens_num)

        return y_hard, y_hard


class BERTIdealEmissionRateCompressionModule(TeacherPruner):

    def __init__(self):
        super().__init__()
        self.register_student_model(BERTStudentPruner(input_units=768))

    def forward(self, attentions, embedding_sequence, compression_rate):
        if self.use_teacher_output:
            remain_tokens_num = int(embedding_sequence.size()[1] * (1 - compression_rate))
            # remain tokens num must be positive.
            remain_tokens_num = max(remain_tokens_num, 1)

            prod = attentions[0].mean(dim=1)
            for attention in attentions[1:]:
                prod *= attention.mean(dim=1)

            y_soft = - prod[:, 0, :]

            y_hard = select_k(y_soft, remain_tokens_num)

            # be careful with following statement.
            # do not train student model when inferencing.
            # otherwise it will cause an unexpected error.
            if self.training:
                self.train_student(
                    inputs=embedding_sequence, expected_out=y_soft, normalize_dim=1
                )
                self.zero_grad()

            return y_hard, y_soft
        else:
            return self.student_forward(embedding_sequence, compression_rate)


class IdealGradientCompressionModule(TeacherPruner):

    def __init__(self):
        super().__init__()
        self.register_student_model(BERTStudentPruner(input_units=768))

    def forward(self, grad, embedding_sequence, compression_rate):
        if self.use_teacher_output:
            remain_tokens_num = int(embedding_sequence.size()[1] * (1 - compression_rate))
            # remain tokens num must be positive.
            remain_tokens_num = max(remain_tokens_num, 1)

            y_soft = - grad.abs().mean(dim=-1)

            y_hard = select_k(y_soft, remain_tokens_num)

            # be careful with following statement.
            # do not train student model when inferencing.
            # otherwise it will cause an unexpected error.
            if self.training:
                self.train_student(
                    inputs=embedding_sequence, expected_out=y_soft, normalize_dim=1
                )
                self.zero_grad()

            return y_hard, y_soft
        else:
            return self.student_forward(embedding_sequence, compression_rate)


PRUNERS = {
    "random": RandomCompressionModule,
    "frequency": FrequencyCompressionModule,
    "head": HeadCompressionModule,
    "tail": TailCompressionModule,
    "lstm": LSTMSelfCompressionModule,
    "ideal emission rate": BERTIdealEmissionRateCompressionModule,
    "ideal gradients":IdealGradientCompressionModule,
}
