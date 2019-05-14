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
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, \
    Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Helper functions

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def initial_boost(epoch):
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


# Data cleaning Functions

def classMerge():
    def read_csv(file, lst, cats):
        with open(file, newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                cat = row[1]
                lst.append(row[0])
                cats.append(int(row[1]))

    data = []
    cats = []
    read_csv("Databases/data.csv", data, cats)

    dataWriter = csv.writer(open('data_merged.csv', 'w'), delimiter=',',
                            lineterminator="\n")
    for n, point in enumerate(data):
        category = cats[n]
        is_happy = category == 3 or category == 5 or category == 6 or \
                   category == 7 or category == 9 or category == 11
        is_neutral = category == 0 or category == 1 or category == 10
        is_sad = category == 2 or category == 4
        if is_happy:
            dataWriter.writerow([point, 1])
        elif is_neutral:
            dataWriter.writerow([point, 0])
        elif is_sad:
            dataWriter.writerow([point, 2])
        elif category == 8:
            dataWriter.writerow([point, 3])
        elif category == 12:
            dataWriter.writerow([point, 4])
        else:
            print(point, n)

    count = 0
    with open('data_merged.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for _ in readCSV:
            count += 1

    print(count)


def dataProcessing():
    # stop_words = list(get_stop_words('en'))         #About 900 stop words
    # nltk_words = list(stopwords.words('english'))   #About 150 stop words
    # stop_words.extend(nltk_words)

    def word_prob(word):
        return dictionary[word] / total

    def words(text):
        return re.findall('[a-z]+', text.lower())

    dictionary = Counter(words(open('Databases/wordlists/merged.txt').read()))
    max_word_length = max(map(len, dictionary))
    total = float(sum(dictionary.values()))

    def viterbi_segment(text):
        probs, lasts = [1.0], [0]
        for i in range(1, len(text) + 1):
            prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                            for j in range(max(0, i - max_word_length), i))
            probs.append(prob_k)
            lasts.append(k)
        words = []
        i = len(text)
        while 0 < i:
            words.append(text[lasts[i]:i])
            i = lasts[i]
        words.reverse()
        return words, probs[-1]

    def fix_hashtag(text):
        text = text.group().split(":")[0]
        text = text[1:]  # remove '#'
        try:
            test = int(text[0])
            text = text[1:]
        except:
            pass
        output = ' '.join(viterbi_segment(text)[0])
        # print(output)
        return output

    def clean_tweet(this_tweet):
        this_tweet = this_tweet.lower()
        this_tweet = re.sub("(#[A-Za-z0-9]+)", fix_hashtag, this_tweet)
        return ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ",
                   this_tweet).split())

    def remove_stopwords(word_list):
        filtered_tweet = ""
        for word in word_list:
            word = word.lower()
            if word not in stopwords.words("english"):
                filtered_tweet = filtered_tweet + " " + word

        return filtered_tweet.lstrip()

    def vectorise_label(label):
        if label == "empty":
            return 1  # neutral
        elif label == "sadness":
            return 2  # sad
        elif label == "enthusiasm":
            return 3  # happy
        elif label == "neutral":
            return 0  # neutral
        elif label == "worry":
            return 4  # sad
        elif label == "surprise":
            return 5  # happy
        elif label == "love":
            return 6  # happy
        elif label == "fun":
            return 7  # happy
        elif label == "hate":
            return 8
        elif label == "happiness":
            return 9  # happy
        elif label == "boredom":
            return 10  # neutral
        elif label == "relief":
            return 11  # happy
        elif label == "anger":
            return 12

    data_train = pd.read_csv('Databases/data/text_emotion.csv', sep=',')
    print("Dataset shape:", data_train.shape)
    print(data_train.sentiment[0], ":", data_train.content[0])

    data_writer = csv.writer(open('Databases/data.csv', 'w'), delimiter=',',
                             lineterminator="\n")

    total = 40000
    for i in range(40000):
        # print("Progress: ",round(i/total*100,2),"   ",end="\r")
        tweet = clean_tweet(data_train.content[i])
        # tweet = remove_stopwords(tweet.split())
        data_writer.writerow(
            [tweet, str(vectorise_label(data_train.sentiment[i]))])
        # sys.stdout.write("\033[F")

    print("Progress: ", 100, "\nComplete!")

    count = 0
    with open('Databases/data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            count += 1


def setup():
    # GLobal Set up variables
    global MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM, \
        GLOVE_DIR, TEXTS, LABELS, TOKENIZER, DATA, SEQ, WORD_INDEX, DATA_INT, \
        DATA, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, EMBEDDING_INDEX, \
        EMBEDDING_MATRIX

    MAX_NB_WORDS = 40000  # max no. of words for tokenizer
    MAX_SEQUENCE_LENGTH = 300  # max length of text (words) including padding
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = 300  # embedding dimensions for word vectors (word2vec/GloVe)
    GLOVE_DIR = "Databases/glove/glove.42B." + str(
        EMBEDDING_DIM) + "d.txt"
    print("[i] Loaded Parameters:\n",
          MAX_NB_WORDS, MAX_SEQUENCE_LENGTH + 5,
          VALIDATION_SPLIT, EMBEDDING_DIM, "\n",
          GLOVE_DIR)

    TEXTS, LABELS = [], []
    print("[i] Reading from csv file...", end="")
    with open('Databases/data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            TEXTS.append(row[0])
            LABELS.append(row[1])
    print("Done!")

    with open('tokenizer.pickle', 'rb') as handle:
        TOKENIZER = pickle.load(handle)

    SEQ = TOKENIZER.texts_to_sequences(TEXTS)
    WORD_INDEX = TOKENIZER.word_index
    print('[i] Found %s unique tokens.' % len(WORD_INDEX))
    DATA_INT = pad_sequences(SEQ, padding='pre',
                             maxlen=(MAX_SEQUENCE_LENGTH - 5))
    DATA = pad_sequences(DATA_INT, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    LABELS = to_categorical(
        np.asarray(LABELS))  # convert to one-hot encoding vectors
    print('[+] Shape of data tensor:', DATA.shape)
    print('[+] Shape of label tensor:', LABELS.shape)

    indices = np.arange(DATA.shape[0])
    np.random.shuffle(indices)
    DATA = DATA[indices]
    LABELS = LABELS[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * DATA.shape[0])
    print('[+] Collecting Partitions:', nb_validation_samples)
    X_TRAIN = DATA[:-nb_validation_samples]
    Y_TRAIN = LABELS[:-nb_validation_samples]
    X_TEST = DATA[-nb_validation_samples:]
    Y_TEST = LABELS[-nb_validation_samples:]
    print('[+] Collected Partitions:')
    EMBEDDING_INDEX = {}
    f = open(GLOVE_DIR, encoding="utf8")
    print("Commencing read")
    print("[i] Loading GloVe from:", GLOVE_DIR, "...", end="")
    for line in f:
        values = line.split()
        word = values[0]
        EMBEDDING_INDEX[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    print("Done.\n[+] Proceeding with Embedding Matrix...", end="")
    EMBEDDING_MATRIX = np.random.random((len(WORD_INDEX) + 1, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = EMBEDDING_INDEX.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            EMBEDDING_MATRIX[i] = embedding_vector
    print("[i] Completed!")


def build_network():
    embedding_matrix_ns = np.random.random((len(WORD_INDEX) + 1, EMBEDDING_DIM))
    for word, i in WORD_INDEX.items():
        embedding_vector = EMBEDDING_INDEX.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_ns[i] = embedding_vector
    print("Completed!")

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

    l_conv_2 = Conv1D(filters=24, kernel_size=2, activation='relu')(l_lstm1)
    l_conv_2 = Dropout(0.3)(l_conv_2)
    l_conv_3 = Conv1D(filters=24, kernel_size=3, activation='relu')(l_lstm1)
    l_conv_3 = Dropout(0.3)(l_conv_3)

    l_conv_5 = Conv1D(filters=24, kernel_size=5, activation='relu', )(l_lstm1)
    l_conv_5 = Dropout(0.3)(l_conv_5)
    l_conv_6 = Conv1D(filters=24, kernel_size=6, activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_6 = Dropout(0.3)(l_conv_6)

    l_conv_8 = Conv1D(filters=24, kernel_size=8, activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_8 = Dropout(0.3)(l_conv_8)

    conv_1 = [l_conv_6, l_conv_5, l_conv_8, l_conv_2, l_conv_3]

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

    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.002)
    lr_metric = get_lr_metric(adadelta)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

    model_checkpoints = callbacks.ModelCheckpoint(
        "checkpoint-{val_loss:.3f}.h5", monitor='val_loss', verbose=0,
        save_best_only=True, save_weights_only=False, mode='auto', period=0)
    lr_schedule = callbacks.LearningRateScheduler(initial_boost)

    model.summary()
    model.save('BalanceNet.h5')

    model = keras.models.load_model("checkpoint-1.097.h5")

    model_log = model.fit(X_TRAIN, Y_TRAIN, validation_data=(X_TEST, Y_TEST),
                          epochs=200, batch_size=128)

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
