import os
import argparse
from copy import deepcopy

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from model.data import SQuAD
from model.model import BiDAF


class MovingAverage:
    def __init__(self, decay_rate=0.999):
        self.decay_rate = decay_rate
        self.weights_dict = dict()

    def init(self, name, val):
        self.weights_dict[name] = val.clone()

    def get(self, name):
        return self.weights_dict[name]

    def update(self, name, val):
        self.weights_dict[name] *= self.decay_rate
        self.weights_dict[name] += (1-self.decay_rate) * val.clone()


def train(data, model, args):
    ma_dict = MovingAverage()  # moving averages of all weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            ma_dict.init(name, param.data)

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
            p1, p2 = model(batch)
            loss = criterion(p1, batch.p_begin) + criterion(p2, batch.p_end)
            batch_loss += loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            iter += 1

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ma_dict.update(name, param.data)

            if iter % args.validation_freq:
                dev_loss, dev_f1, dev_em = validation(data, model, ma_dict, args)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_dev_exact = dev_em
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


def validation(data, model, ma_dict, args):
    model.eval()

    for batch in iter(data.dev_iter):
        p1, p2 = model(batch)


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

    data = SQuAD(squad_version=args.squad_version,
                 word_vec_dim=args.word_vec_dim,
                 train_batch_size=args.train_batch_size,
                 dev_batch_size=args.dev_batch_size,
                 gpu=args.gpu)
    setattr(args, 'char_vocab_size', len(data.CHAR_NESTING.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(pretrain_embedding=data.WORD.vocab.vectors,
                  char_vocab_size=len(data.CHAR_NESTING.vocab),
                  hidden_size=args.hidden_size,
                  char_emb_dim=args.char_emb_dim,
                  char_channel_num=args.char_channel_num,
                  char_channel_width=args.char_channel_width,
                  dropout=args.dropout
                  ).to(device)

    trained_model = train(data, model, args)
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(trained_model.state_dict(), f'models/BiDAF_{args.model_time}.pt')


if __name__ == "__main__":
    main()
