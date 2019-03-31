import os
import argparse
from copy import deepcopy

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

import evaluate
from model.data import SQuAD
from model.model import BiDAF


class WeightDict:
    def __init__(self, ):
        self.weights_dict = dict()

    def put(self, name, val):
        self.weights_dict[name] = val.clone()

    def get(self, name):
        return self.weights_dict[name]

    def ema_update(self, name, val, decay_rate=0.999):
        self.weights_dict[name] *= decay_rate
        self.weights_dict[name] += (1-decay_rate) * val.clone()


def train(device, model, data, args):
    model = model.to(device)
    weight_dict = WeightDict()  # moving averages of all weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_dict.put(name, param.data)

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    writer = SummaryWriter('log')
    best_dev_f1 = 0
    best_dev_em = 0
    iter = 0
    for i_epoch in range(args.epoch):
        print(f'Epoch {i_epoch}')
        data.train_iter.init_epoch()
        epoch_loss = 0.0

        batch_loss = 0.0
        for i, batch in enumerate(data.train_iter):
            optimizer.zero_grad()
            p1, p2 = model(batch)
            loss = criterion(p1, batch.p_begin) + criterion(p2, batch.p_end)
            batch_loss += loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            iter += 1

            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_dict.ema_update(name, param.data, decay_rate=args.moving_average_decay)

            if iter % args.validation_freq:
                model.eval()
                dev_loss, dev_f1, dev_em = validation(device, model, data, weight_dict)
                model.train()
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_dev_em = dev_em
                    best_model = deepcopy(model)

                writer.add_scalar('Dev/Loss', dev_loss, iter)
                writer.add_scalar('Dev/EM', dev_em, iter)
                writer.add_scalar('Dev/F1', dev_f1, iter)
                writer.add_scalar('Train/Loss', batch_loss, iter)
                batch_loss = 0.0

        print(f"Total epoch loss {epoch_loss}")

    writer.close()
    print(f'Best model dev F1: {best_dev_f1:.3f}, max dev EM: {best_dev_em:.3f}')
    return best_model


def validation(device, model, data, weight_dict):
    backup_weight_dict = WeightDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_weight_dict.put(name, param.data)
            param.data.copy_(weight_dict.get(name))

    predictions = dict()
    criterion = nn.CrossEntropyLoss()
    dev_loss = 0.0
    for batch in iter(data.dev_iter):
        p1, p2 = model(batch)
        loss = criterion(p1, batch.p_begin) + criterion(p2, batch.p_end)
        dev_loss += loss.item()

        batch_size, x_len = p1.size()
        mask = torch.triu(torch.ones(x_len, x_len).to(device)).unsqueeze(0).expand(batch_size, -1, -1)
        prob = p1.unsqueeze(-1) * p2.unsqueeze(-2) * mask
        prob, e_idx = prob.max(dim=2)
        prob, s_idx = prob.max(dim=1)

        for i in range(batch_size):
            id = batch.id[i]
            p_begin = s_idx[i].item()
            p_end = e_idx[i][p_begin].item()
            answer = batch.x_word[0][i][p_begin:p_end + 1]
            answer = ' '.join([data.WORD.vocab.itos[idx] for idx in answer])
            predictions[id] = answer

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_weight_dict.get(name))

    eval = evaluate.evaluate(data.dev_set, predictions)
    return dev_loss, eval['f1'], eval['exact_match']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-emb-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-num', default=100, type=int)
    parser.add_argument('--dev-batch-size', default=60, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--moving-average-decay', default=0.999, type=float)
    parser.add_argument('--squad-version', default='1.1')
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--word-vec-dim', default=100, type=int)
    parser.add_argument('--validation-freq', default=100, type=int)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Load SQuAD {args.squad_version}")
    data = SQuAD(device=device,
                 squad_version=args.squad_version,
                 word_vec_dim=args.word_vec_dim,
                 train_batch_size=args.train_batch_size,
                 dev_batch_size=args.dev_batch_size)
    setattr(args, 'char_vocab_size', len(data.CHAR_NESTING.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))

    print(f"Load BiDAF Model")
    model = BiDAF(pretrain_embedding=data.WORD.vocab.vectors,
                  char_vocab_size=len(data.CHAR_NESTING.vocab),
                  hidden_size=args.hidden_size,
                  char_emb_dim=args.char_emb_dim,
                  char_channel_num=args.char_channel_num,
                  char_channel_width=args.char_channel_width,
                  dropout=args.dropout
                  )

    trained_model = train(device, model, data, args)
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(trained_model.state_dict(), f'models/BiDAF_{args.model_time}.pt')


if __name__ == "__main__":
    main()
