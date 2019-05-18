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



def get_max_checkpoint(tok):
    model_dir = os.listdir('.') if tok else os.listdir("../")
    prefix = "" if tok else "../"
    fnames = sorted([(float(fname[fname.find("-")+1:
                            fname.find(".", fname.find("-")+3)]), prefix+fname)
                     for fname in model_dir
                     if fname.startswith("checkpoint_val_acc-")])
    print(fnames)
    return fnames[-1]



#The global constants below are designed for sentiment analysis
def prep_sent(tok=None):
    global MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM, \
        TOKENIZER, CLASSES, MODEL_TEST
    MAX_NB_WORDS = 40000  # max no. of words for tokenizer
    MAX_SEQUENCE_LENGTH = 30  # max length of text (words) including padding
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = 300  # embedding dimensions for word vectors (word2vec/GloVe)

    # Open our word tokenizer
    f = open('tokenizer.pickle', 'rb') if tok else open('../tokenizer.pickle', 'rb')
    TOKENIZER = pickle.load(f)
    K.clear_session()
    CLASSES = ["neutral", "happy", "sad", "hate", "anger"]
    best_model = get_max_checkpoint(tok)
    MODEL_TEST = load_model(best_model[1])
    MODEL_TEST._make_predict_function()




def get_sent(line):
    """ Given an input line runs sentiment
    analysis model to determine the relative
    Neutrality, Happiness, Angriness, Sadness
    and Hate emotions of speech
    """
    parsedL = []
    parsedL.append(line)
    sequences_test = TOKENIZER.texts_to_sequences(parsedL)

    data_int_t = pad_sequences(sequences_test, padding='pre',
                               maxlen=(MAX_SEQUENCE_LENGTH - 5))
    data_test = pad_sequences(data_int_t, padding='post',
                              maxlen=MAX_SEQUENCE_LENGTH)

    y_prob = MODEL_TEST.predict(data_test)
    print(y_prob)
    return y_prob

def main(line, tok):
    prep_sent(tok)
    print(get_sent(line))

if __name__ == '__main__':
    defaultLine = ""
    line = sys.argv[1] if len(sys.argv) > 1 else defaultLine
    main(line, "tokenizer.pickle")