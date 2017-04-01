import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    # ******** original ********
    # def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):

    # ******** marco ********
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):

        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(19443, 200)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        # ******** original ********

        # self.init_weights()

        # ******** marco ********

        self.init_weights()

        # ***********************

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    # ******** original ********

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.fill_(0)
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    # ******** marco ******** (taken rom sentiment code)

    def init_weights(self):
        initrange = 0.1
        # print(pretrained)
        # if pretrained:
        print("Setting Pretrained Embeddings")

        # ******** original ********
        # pretrained = pretrained.astype(np.float32)

        # ******** marco ********
        # pretrained = np.loadtxt("../vectors/_glove.6B.50d_sampled_vuacm_voc_no_words_new_version_with_random_vector.txt", delimiter=" ")
        pretrained = np.loadtxt("../vectors/vectors/vectors.LiverpoolFC.txt", delimiter=" ")
        # pretrained = torch.from_numpy(pretrained)

        # if(self.args.cuda):
        #     pretrained = pretrained.cuda()
        # self.encoder.weight.data = pretrained
        self.encoder.weight.data.copy_(torch.from_numpy(pretrained)) # see here: https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222

        # else:
        #     print("No Pretrained Embeddings")
        #     self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # ***********************



    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
