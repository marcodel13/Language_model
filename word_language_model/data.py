import os
import torch
import numpy as np


class Dictionary(object):
    def __init__(self, file_path=None):
        self.word2idx = {}
        self.idx2word = []

        if file_path:
            self.load_from_file(file_path)

    def load_from_file(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                word = line.split()[0]
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
#
# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = []
#
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, dic_file_path):
        self.dic_file_path = dic_file_path
        self.dictionary = Dictionary(dic_file_path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary


        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if not self.dic_file_path:
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        # RADOM CHECK
                        # rand_indx = np.random.randint(0,len(self.dictionary.idx2word))
                        # ids[token] = rand_indx
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        ids[token] = self.dictionary.word2idx['unk']
                    token += 1

        return ids
