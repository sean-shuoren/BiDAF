import io
import json
import os

from torchtext import data
from torchtext.vocab import GloVe
import spacy


class SQuAD(object):
    def __init__(self, device, squad_version="1.1", word_vec_dim=100, train_batch_size=60, dev_batch_size=60):
        self.train_file = f'train-v{squad_version}.json'
        self.dev_file = f'dev-v{squad_version}.json'
        self.raw_dir = os.path.join('data', 'raw')
        self.processed_dir = os.path.join('data', 'processed')

        # Prepocess json to json list
        self.spacy = spacy.load('en')
        if not os.path.exists(os.path.join(self.processed_dir, self.train_file)):
            self.pre_process(self.raw_dir, self.train_file, self.processed_dir)
        if not os.path.exists(os.path.join(self.processed_dir, self.dev_file)):
            self.pre_process(self.raw_dir, self.dev_file, self.processed_dir)

        # Load data using torchtext
        self.ID = data.RawField()
        self.CHAR_NESTING = data.Field(batch_first=True, lower=True, tokenize=list)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=self.tokenizer)
        self.WORD = data.Field(batch_first=True, include_lengths=True, lower=True, tokenize=self.tokenizer)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)
        self.fields = {'id': ('id', self.ID),
                       'context': [('x_word', self.WORD), ('x_char', self.CHAR)],
                       'query': [('q_word', self.WORD), ('q_char', self.CHAR)],
                       'p_begin': ('p_begin', self.LABEL),
                       'p_end': ('p_end', self.LABEL)}
        self.train, self.dev = data.TabularDataset.splits(path=self.processed_dir,
                                                          train=self.train_file,
                                                          validation=self.dev_file,
                                                          format='json',
                                                          fields=self.fields)
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=word_vec_dim))
        self.train_iter, self.dev_iter = data.BucketIterator.splits(
            (self.train, self.dev),
            batch_sizes=[train_batch_size, dev_batch_size],
            device=device,
            sort_key=lambda x: len(x.x_word),
            sort_within_batch=True)

        # Pre-load devset for validation
        dev_set_file = open(os.path.join(self.raw_dir, self.dev_file))
        self.validation_dev_set = json.load(dev_set_file)['data']

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
                        id = qa['id']
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
                            out.append(dict([('id', id),
                                             ('context', context),
                                             ('query', question),
                                             ('answer', ans['text']),
                                             ('p_begin', p_begin),
                                             ('p_end', p_end)]))

        out_filename = os.path.join(output_dir, input_file)
        with open(out_filename, 'w', encoding='utf-8') as f:
            for o in out:
                json.dump(o, f)
                f.write('\n')
