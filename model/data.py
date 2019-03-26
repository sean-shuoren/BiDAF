import io
import json
import os

from torchtext import data
from torchtext.vocab import GloVe
import spacy


class SQuAD(object):
    def __init__(self, squad_version="1.1", word_vec_dim=100, train_batch_size=60, dev_batch_size=60, gpu=0):
        train_file = f'train-v{squad_version}.json'
        dev_file = f'dev-v{squad_version}.json'
        raw_dir = os.path.join('data', 'raw')
        processed_dir = os.path.join('data', 'processed')
        if not os.path.exists(os.path.join(processed_dir, train_file)):
            self.pre_process(raw_dir, train_file, processed_dir)
        if not os.path.exists(os.path.join(processed_dir, dev_file)):
            self.pre_process(raw_dir, dev_file, processed_dir)

        self.spacy = spacy.load('en')
        self.CHAR_NESTING = data.Field(batch_first=True, lower=True, tokenize=list)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=self.tokenizer)
        self.WORD = data.Field(batch_first=True, include_lengths=True, lower=True, tokenize=self.tokenizer)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'context': [('x_word', self.WORD), ('x_char', self.CHAR)],
                       'query': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'p_begin': ('p_begin', self.LABEL),
                       'p_end': ('p_end', self.LABEL)}

        train, dev = data.TabularDataset.splits(path=processed_dir,
                                                train=train_file,
                                                validation=dev_file,
                                                format='json',
                                                fields=dict_fields)

        self.CHAR.build_vocab(train, dev)
        self.WORD.build_vocab(train, dev, vectors=GloVe(name='6B', dim=word_vec_dim))
        self.train_iter, self.dev_iter = data.BucketIterator.splits(
            (train, dev),
            batch_sizes=[train_batch_size, dev_batch_size],
            device=gpu,
            sort_key=lambda x: len(x.c_word))

    def tokenizer(self, text):
        return [t.text for t in self.spacy.tokenizer(text)]

    def pre_process(self, input_dir, input_file, output_dir):
        in_filename = os.path.join(input_dir, input_file)
        out = []
        with io.open(in_filename, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)['data']
            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = self.tokenizer(context)
                    for qa in paragraph['qas']:
                        question = qa['question']
                        for ans in qa['answers']:
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(ans['text'])
                            cum_len = 0
                            p_begin = -1
                            p_end = -1
                            answer = ""
                            for i, t in enumerate(tokens):
                                while context[cum_len] == ' ':
                                    cum_len += 1
                                if p_begin == -1 and s_idx <= cum_len:
                                    p_begin = i
                                if p_begin != -1:
                                    if len(answer) > 0:
                                        answer += ' '
                                    answer += t
                                cum_len += len(t)
                                if p_end == -1 and e_idx <= cum_len:
                                    p_end = i
                                    if p_begin == -1:
                                        p_begin = i
                                    break
                            assert p_begin != -1 and p_end != -1
                            out.append(dict([('context', context),
                                             ('query', question),
                                             ('answer', answer),
                                             ('p_begin', p_begin),
                                             ('p_end', p_end)]))

        out_filename = os.path.join(output_dir, input_file)
        with open(out_filename, 'w', encoding='utf-8') as f:
            for o in out:
                json.dump(o, f)
                f.write('\n')
