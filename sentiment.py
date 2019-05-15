import re, sys, os, csv, pickle
import re
import os
from keras import regularizers, initializers, optimizers, callbacks
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

#The global constants below are designed for sentiment analysis

MAX_NB_WORDS = 40000  # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30  # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200  # embedding dimensions for word vectors (word2vec/GloVe)

# Open our word tokenizer
f = open('tokenizer.pickle', 'rb')
tokenizer = pickle.load(f)
K.clear_session()
classes = ["neutral", "happy", "sad", "hate", "anger"]
model_test = load_model('checkpoint-1.097.h5')

model_test._make_predict_function()

def get_sent(line):
    """ Given an input line runs sentiment
    analysis model to determine the relative
    Neutrality, Happiness, Angriness, Sadness
    and Hate emotions of speech
    """
    parsedL = []
    parsedL.append(line)
    sequences_test = tokenizer.texts_to_sequences(parsedL)

    data_int_t = pad_sequences(sequences_test, padding='pre',
                               maxlen=(MAX_SEQUENCE_LENGTH - 5))
    data_test = pad_sequences(data_int_t, padding='post',
                              maxlen=MAX_SEQUENCE_LENGTH)

    y_prob = model_test.predict(data_test)
    return y_prob

def main(line):
    print(get_sent(line))

if __name__ == '__main__':
    main(sys.argv[1])