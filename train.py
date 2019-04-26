import os
import time
import argparse, ast
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import evaluate, evaluate2
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


def train(device, data, model, epoch=12, lr=0.5, moving_average_decay=0.999, validation_freq=500):
    model = model.to(device)

    weight_dict = WeightDict()  # moving averages of all weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_dict.put(name, param.data)

    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    writer = SummaryWriter('log')
    best_dev_f1 = 0
    best_dev_em = 0
    iteration = 0
    last_iter = 0
    for i_epoch in range(epoch):
        print(f'Epoch {i_epoch}')
        data.train_iter.init_epoch()
        epoch_loss = 0.0

        batch_loss = 0.0
        for batch in iter(data.train_iter):
            optimizer.zero_grad()
            p1, p2 = model(batch)
            loss = criterion(p1, batch.p_begin) + criterion(p2, batch.p_end)
            loss.backward()
            optimizer.step()

            iteration += 1
            batch_loss += loss.item()
            epoch_loss += loss.item()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_dict.ema_update(name, param.data, decay_rate=moving_average_decay)

            if iteration % validation_freq == 0:
                model.eval()
                with torch.no_grad():
                    dev_loss, dev_f1, dev_em = validation(device=device,
                                                          data=data,
                                                          model=model,
                                                          weight_dict=weight_dict)
                model.train()
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_dev_em = dev_em
                    best_model = deepcopy(model)
                writer.add_scalar('Model/Bias1', model.na_bias_1.item(), iteration)
                writer.add_scalar('Model/Bias2', model.na_bias_2.item(), iteration)
                writer.add_scalar('Dev/Loss', dev_loss, iteration)
                writer.add_scalar('Dev/EM', dev_em, iteration)
                writer.add_scalar('Dev/F1', dev_f1, iteration)
                writer.add_scalar('Train/Loss', batch_loss/(iteration-last_iter), iteration)
                print(f"Iteration {iteration}, Train Loss {batch_loss}, Dev Loss {dev_loss}")
                batch_loss = 0.0
                last_iter = iteration

        print(f"Total epoch loss {epoch_loss}")

    writer.close()
    print(f'Best model dev F1: {best_dev_f1:.3f}, max dev EM: {best_dev_em:.3f}')
    return best_model


def validation(device, data, model, weight_dict):
    backup_weight_dict = WeightDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_weight_dict.put(name, param.data)
            param.data.copy_(weight_dict.get(name))

    predictions = dict()
    if data.version == "2.0":
        na_probs = dict()
    criterion = nn.CrossEntropyLoss()
    dev_loss = 0.0
    for batch in iter(data.dev_iter):
        p1, p2 = model(batch)
        loss = criterion(p1, batch.p_begin) + criterion(p2, batch.p_end)
        dev_loss += loss.item()

        # Prepare answers
        batch_size, x_len = p1.size()
        if data.version == "1.1":
            mask = torch.triu(torch.ones(x_len, x_len))
        else:
            mask = torch.cat([torch.triu(torch.ones(x_len, x_len-1)), torch.zeros(x_len, 1)], dim=-1)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        mask = mask.to(device)

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
            if data.version == "2.0":
                na_probs[id] = F.softmax(p1[i], dim=-1)[-1].item() * F.softmax(p2[i], dim=-1)[-1].item()

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(backup_weight_dict.get(name))

    if data.version == "1.1":
        eval = evaluate.evaluate(data.validation_dev_set, predictions)
    elif data.version == "2.0":
        eval = evaluate2.evaluate(data.validation_dev_set, predictions, na_probs)
    return dev_loss, eval['f1'], eval['exact']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-emb-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-num', default=100, type=int)
    parser.add_argument('--dev-batch-size', default=60, type=int)
    parser.add_argument('--disable-c2q', type=ast.literal_eval)
    parser.add_argument('--disable-q2c', type=ast.literal_eval)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--moving-average-decay', default=0.999, type=float)
    parser.add_argument('--squad-version', default='1.1')
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--word-vec-dim', default=100, type=int)
    parser.add_argument('--validation-freq', default=500, type=int)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Device is set to", device)

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
                  consider_na=True if args.squad_version == "2.0" else False,
                  enable_c2q= not args.disable_c2q if args.disable_c2q is not None else True,
                  enable_q2c= not args.disable_q2c if args.disable_q2c is not None else True,
                  hidden_size=args.hidden_size,
                  char_emb_dim=args.char_emb_dim,
                  char_channel_num=args.char_channel_num,
                  char_channel_width=args.char_channel_width,
                  dropout=args.dropout)

    print(f"Training start")
    trained_model = train(device=device,
                          data=data,
                          model=model,
                          epoch=args.epoch,
                          lr=args.learning_rate,
                          moving_average_decay=args.moving_average_decay,
                          validation_freq=args.validation_freq)
    model_time = int(time.time())
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(trained_model.state_dict(), f'trained_models/BiDAF_{model_time}.pt')


if __name__ == "__main__":
    main()
