# Build the Sentiment analysis model, this takes a looooong time, so we
# save the model and reuse the weights rather than retraining every time


# import statements
import csv
import numpy as np
import pandas as pd
from many_stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re, sys, os, csv, keras, pickle, itertools
import tensorflow as tf
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, \
    Bidirectional
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.engine.topology import Layer, InputSpec
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Helper functions
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def boost(epoch):
    if epoch == 0:
        return float(8.0)
    elif epoch == 1:
        return float(4.0)
    elif epoch == 2:
        return float(2.0)
    elif epoch == 3:
        return float(1.5)
    else:
        return float(1.0)


def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch % 33 == 0:
            multiplier = 10
        else:
            multiplier = 1
        rate = float(multiplier * l_r * 1 / (1 + decay * epoch))
        # print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:", str(e))
        return float(1.0)


# Model Building Functions

# Reads in and organizes data for NN, partitions data set,
# initialize core architectural components
def setup():
    # Global variables
    global MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM, \
        GLOVE_DIR, TEXTS, LABELS, TOKENIZER, DATA, SEQ, WORD_INDEX, DATA_INT, \
        DATA, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, EMBEDDING_INDEX, \
        EMBEDDING_MATRIX

    # Make sure we're using tensorflow on GPU
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    MAX_NB_WORDS = 40000  # Max # of words for tokenizer
    MAX_SEQUENCE_LENGTH = 30  # Max length of text (words) including padding
    VALIDATION_SPLIT = 0.2  # Validation partition ratio
    EMBEDDING_DIM = 300  # Embedding dimension for word vectors (word2vec/GloVe)
    # Stanford Global Vectors for word representation
    GLOVE_DIR = "Databases/glove/glove.42B." + str(
        EMBEDDING_DIM) + "d.txt"

    print("Loaded Parameters:\n",
          MAX_NB_WORDS, MAX_SEQUENCE_LENGTH + 5,
          VALIDATION_SPLIT, EMBEDDING_DIM, "\n",
          GLOVE_DIR)
    # Load words and emotions
    TEXTS, LABELS = [], []
    print("Reading from csv file...", end="")
    with open('Databases/data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            TEXTS.append(row[0])
            LABELS.append(row[1])
    print("Done!")

    # Load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        TOKENIZER = pickle.load(handle)

    SEQ = TOKENIZER.texts_to_sequences(TEXTS)
    print("SEQ shape", len(SEQ))
    WORD_INDEX = TOKENIZER.word_index
    print('Unique tokens:', len(WORD_INDEX))
    DATA_INT = pad_sequences(SEQ, padding='pre',
                             maxlen=(MAX_SEQUENCE_LENGTH - 5))
    DATA = pad_sequences(DATA_INT, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    LABELS = to_categorical(
        np.asarray(LABELS))  # convert to one-hot encoding vectors
    print('Shape of data:', DATA.shape)
    print('Shape of label:', LABELS.shape)

    # Partition the data
    indices = np.arange(DATA.shape[0])
    np.random.shuffle(indices)
    DATA = DATA[indices]
    LABELS = LABELS[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * DATA.shape[0])
    print('Collecting Partitions:', nb_validation_samples)
    X_TRAIN = DATA[:-nb_validation_samples]
    Y_TRAIN = LABELS[:-nb_validation_samples]
    X_TEST = DATA[-nb_validation_samples:]
    Y_TEST = LABELS[-nb_validation_samples:]
    print("X_Train shape", X_TRAIN.shape, "Y_Train shape", Y_TRAIN.shape,
          "X_Test shape", X_TEST.shape, "Y_Test shape", Y_TEST.shape)
    print('Collected Partitions:')
    EMBEDDING_INDEX = {}
    f = open(GLOVE_DIR, encoding="utf8")
    print("Commencing read")
    print("Loading GloVe from:", GLOVE_DIR, "...", end="")
    for line in f:
        values = line.split()
        word = values[0]
        EMBEDDING_INDEX[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    print("Done.\n[+] Proceeding with Embedding Matrix...", end="")

    # Create embedding layer matrix
    EMBEDDING_MATRIX = np.random.random((len(WORD_INDEX) + 1, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = EMBEDDING_INDEX.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            EMBEDDING_MATRIX[i] = embedding_vector
    print("Completed Setup!")


# Put together the actual layout of the neural net, and train and test it
def build_network():
    # Without this tensorflow doesn't play nice on GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow GPU memory
    config.log_device_placement = True  # to log device placement
    sess = tf.Session(config=config)
    set_session(
        sess)  # set this TensorFlow session as the default session for Keras

    # Create a second embedding layer matrix, this one is dynamic, the other is
    # static, which allows for learning specific embeddings, while also
    # retaining a generalized embedding and using whichever is better
    embedding_matrix_ns = np.random.random((len(WORD_INDEX) + 1, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = EMBEDDING_INDEX.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_ns[i] = embedding_vector
    print("Embedding matrix 2 Completed!")

    # Input is only 30 words at a time, while we originally tried processing the
    # entire text at once, our accuracy sharply declined
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # static channel
    embedding_layer_frozen = Embedding(len(WORD_INDEX) + 1,
                                       EMBEDDING_DIM,
                                       weights=[EMBEDDING_MATRIX],
                                       input_length=MAX_SEQUENCE_LENGTH,
                                       trainable=False)
    embedded_sequences_frozen = embedding_layer_frozen(sequence_input)

    # non-static channel
    embedding_layer_train = Embedding(len(WORD_INDEX) + 1,
                                      EMBEDDING_DIM,
                                      weights=[embedding_matrix_ns],
                                      input_length=MAX_SEQUENCE_LENGTH,
                                      trainable=True)
    embedded_sequences_train = embedding_layer_train(sequence_input)

    # First Half: Long Short-Term Memory > Convolutional Neural Net
    l_lstm1f = Bidirectional(
        LSTM(6, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(
        embedded_sequences_frozen)
    l_lstm1t = Bidirectional(
        LSTM(6, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(
        embedded_sequences_train)
    l_lstm1 = Concatenate(axis=1)([l_lstm1f, l_lstm1t])

    # Build CNN to attach to LSTM
    l_conv_2 = Conv1D(filters=24, kernel_size=2, activation='relu')(l_lstm1)
    l_conv_2 = Dropout(0.3)(l_conv_2)
    l_conv_3 = Conv1D(filters=24, kernel_size=3, activation='relu')(l_lstm1)
    l_conv_3 = Dropout(0.3)(l_conv_3)
    l_conv_4 = Conv1D(filters=24, kernel_size=5, activation='relu', )(l_lstm1)
    l_conv_4 = Dropout(0.3)(l_conv_4)
    l_conv_5 = Conv1D(filters=24, kernel_size=6, activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_5 = Dropout(0.3)(l_conv_5)
    l_conv_6 = Conv1D(filters=24, kernel_size=8, activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_6 = Dropout(0.3)(l_conv_6)

    conv_1 = [l_conv_5, l_conv_4, l_conv_6, l_conv_2, l_conv_3]

    l_lstm_c = Concatenate(axis=1)(conv_1)

    # Second half: Convolution Neural Net > Long Short-Term Memory
    l_conv_4f = Conv1D(filters=12, kernel_size=4, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(
        embedded_sequences_frozen)
    l_conv_4f = Dropout(0.3)(l_conv_4f)
    l_conv_4t = Conv1D(filters=12, kernel_size=4, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(
        embedded_sequences_train)
    l_conv_4t = Dropout(0.3)(l_conv_4t)
    l_conv_3f = Conv1D(filters=12, kernel_size=3, activation='relu', )(
        embedded_sequences_frozen)
    l_conv_3f = Dropout(0.3)(l_conv_3f)
    l_conv_3t = Conv1D(filters=12, kernel_size=3, activation='relu', )(
        embedded_sequences_train)
    l_conv_3t = Dropout(0.3)(l_conv_3t)
    l_conv_2f = Conv1D(filters=12, kernel_size=2, activation='relu')(
        embedded_sequences_frozen)
    l_conv_2f = Dropout(0.3)(l_conv_2f)
    l_conv_2t = Conv1D(filters=12, kernel_size=2, activation='relu')(
        embedded_sequences_train)
    l_conv_2t = Dropout(0.3)(l_conv_2t)
    conv_2 = [l_conv_4f, l_conv_4t, l_conv_3f, l_conv_3t, l_conv_2f, l_conv_2t]

    # LSTM into CNN
    l_merge_2 = Concatenate(axis=1)(conv_2)
    l_c_lstm = Bidirectional(
        LSTM(12, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(
        l_merge_2)

    # Merge the 2 halves
    l_merge = Concatenate(axis=1)([l_lstm_c, l_c_lstm])
    l_pool = MaxPooling1D(4)(l_merge)
    l_drop = Dropout(0.5)(l_pool)
    l_flat = Flatten()(l_drop)
    l_dense = Dense(26, activation='relu')(l_flat)
    preds = Dense(5, activation='softmax')(l_dense)

    # OPtimize and store the model for later benchmarking
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.002)
    lr_metric = get_lr_metric(adadelta)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

    # Store the model as a benchmark
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                        batch_size=16, write_grads=True,
                                        write_graph=True)
    model_checkpoints = callbacks.ModelCheckpoint(
        "checkpoint-{val_loss:.3f}.h5", monitor='val_loss', verbose=0,
        save_best_only=True, save_weights_only=False, mode='auto', period=0)
    lr_schedule = callbacks.LearningRateScheduler(boost)

    model.summary()
    model.save('CNNN_LSTM.h5')

    model = keras.models.load_model("checkpoint-1.097.h5")

    model_log = model.fit(X_TRAIN, Y_TRAIN, validation_data=(X_TEST, Y_TEST),
                          batch_size=512, epochs=200,
                          callbacks=[tensorboard, model_checkpoints])

    pd.DataFrame(model_log.history).to_csv("history-balance.csv")

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    classes = ["neutral", "happy", "sad", "hate", "anger"]

    model_test = load_model('best_weights.h5')
    Y_test = np.argmax(Y_TEST, axis=1)  # Convert one-hot to index
    y_pred = model_test.predict(X_TEST)
    y_pred_class = np.argmax(y_pred, axis=1)
    cnf_matrix = confusion_matrix(Y_test, y_pred_class)

    print(classification_report(Y_test, y_pred_class, target_names=classes))

    def plot_confusion_matrix(cm, labels,
                              normalize=True,
                              title='Confusion Matrix (Validation Set)',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            # print('Confusion matrix, without normalization')
            pass

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, k, format(cm[k, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[k, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.figure(figsize=(20, 10))
    plot_confusion_matrix(cnf_matrix, labels=classes)

    # precision = true_pos / (true_pos + false_pos)
    # recall = true_pos / (true_pos + false_neg)

    test_text = []
    sequences_test = tokenizer.texts_to_sequences(test_text)
    data_int_t = pad_sequences(sequences_test, padding='pre',
                               maxlen=(MAX_SEQUENCE_LENGTH - 5))
    data_test = pad_sequences(data_int_t, padding='post',
                              maxlen=MAX_SEQUENCE_LENGTH)
    y_prob = model.predict(data_test)
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
        print(test_text[n], "\nPrediction:", classes[pred], "\n")


def main():
    setup()
    build_network()


if __name__ == '__main__':
    main()
