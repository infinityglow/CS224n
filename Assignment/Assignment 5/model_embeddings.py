#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch
from cnn import CNN
from highway import Highway
from vocab import VocabEntry

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab, hidden_size, dropout_size=0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_size = dropout_size

        self.char_embed_size = 50
        self.word_embed_size = embed_size
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(self.char_embed_size, self.word_embed_size, 5)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(self.dropout_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_length, batch_size, max_word_length = input.size()

        x_embedded = self.embeddings(input)
        x_reshape = x_embedded.permute([0, 1, 3, 2]).view(-1, self.char_embed_size, max_word_length)
        x_conv = self.cnn(x_reshape)
        x_highway = self.highway(x_conv)
        x_dropout = self.dropout(x_highway)
        output = x_dropout.view([sentence_length, batch_size, self.word_embed_size])
        return output

        ### END YOUR CODE

# vocab = VocabEntry()
# sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'], ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
# input = vocab.to_input_tensor_char(sentences, 'cpu')
# embed = ModelEmbeddings(256, vocab)

