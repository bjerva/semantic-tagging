#!/usr/bin/env python

'''
Semantic tagger

Run with python -O to skip debugging printouts.
Run with python -u to make sure slurm logs are written instantly.
'''

import numpy as np
import random
random.seed(1337)
np.random.seed(1337)  # Freeze seeds for reproducibility

# Keras
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, Input, merge, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.engine.topology import Merge
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ProgbarLogger

# Standard
import os
import argparse
from collections import defaultdict

# System
import utils
import data_utils
#from analysis import write_confusion_matrix, prepare_error_analysis
from config import *

# def read_data(read_data_func, word_to_id=None, word_vectors=None, max_sent_len=None, max_word_len=None):
#     '''
#     Wrapper for data reading.
#     Expects a function for data reading as first argument.
#     TODO: hdf5 data caching
#     '''
#     tag_to_id = defaultdict(lambda: len(tag_to_id))
#     tag_to_id[SENT_START]
#     tag_to_id[SENT_END]
#
#     (X_train, y_train) = read_data_func(args.train, word_to_id, tag_to_id, word_vectors, max_sent_len)
#     (X_dev, y_dev) = read_data_func(args.dev, word_to_id, tag_to_id, word_vectors, max_sent_len)
#
#     if args.test:
#         (X_test, y_test) = read_data_func(args.test, word_to_id, tag_to_id, word_vectors, max_sent_len)
#     else:
#         (X_test, y_test) = None, None
#
#     return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def build_model():
    '''
    Build a Keras model with the functional API
    '''

    if args.aux:
        bn_mode = 2
    else:
        bn_mode = 1

    if args.chars:
        char_input = Input(shape=(args.max_sent_len, args.max_word_len), dtype='int32', name='char_input')

        x = Reshape((args.max_sent_len*args.max_word_len, ))(char_input)
        x = Embedding(char_vocab_size, args.char_embedding_dim, input_length=args.max_sent_len*args.max_word_len)(x)
        x = Reshape((args.max_sent_len, args.max_word_len, args.char_embedding_dim))(x)

        if args.bypass:
            # Let's try to make a 'word' representation using the chars...

            #orig_char_embedding = Reshape((args.max_sent_len, args.max_word_len * args.char_embedding_dim))(x)

            bypassed_char_rep = Convolution2D(args.max_sent_len, 1, args.char_embedding_dim, activation='relu', border_mode='same')(x)
            bypassed_char_rep = AveragePooling2D(pool_size=(args.max_word_len, 1), border_mode='same')(bypassed_char_rep)

            bypassed_char_rep = Reshape((args.max_sent_len, args.char_embedding_dim))(bypassed_char_rep)

            #l = GRU(output_dim=int(args.rnn_dim), return_sequences=True, dropout_W=0.1, dropout_U=0.1, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu', consume_less='cpu')(bypassed_char_rep)
            #l = GRU(output_dim=int(args.rnn_dim/2), return_sequences=True, dropout_W=0.1, dropout_U=0.1, activation='relu', consume_less='cpu')(l)

            #r = GRU(output_dim=int(args.rnn_dim), return_sequences=True, go_backwards=True, dropout_W=0.1, dropout_U=0.1, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu', consume_less='cpu')(bypassed_char_rep)
            #r = GRU(output_dim=int(args.rnn_dim/2), return_sequences=True, go_backwards=True, dropout_W=0.1, dropout_U=0.1, activation='relu', consume_less='cpu')(r)

            #bypassed_char_rep = merge([l, r], mode='concat')


        if args.inception:
            for inc_idx in range(args.inception):
                if args.bn:
                    x = BatchNormalization(mode=bn_mode)(x)

                if args.dropout:
                    x = Dropout(args.dropout)(x)

                one_x_one = Convolution2D(args.max_sent_len, 1, 1, activation='relu', border_mode='same')(x)
                one_x_one = MaxPooling2D(pool_size=(2, 2), border_mode='same')(one_x_one)

                three_x_three = Convolution2D(int(args.max_sent_len/4), 1, 1, activation='relu', border_mode='same')(x)
                three_x_three = Convolution2D(args.max_sent_len, 3, 3, activation='relu', border_mode='same')(three_x_three)
                three_x_three = MaxPooling2D(pool_size=(2, 2), border_mode='same')(three_x_three)

                five_x_five = Convolution2D(int(args.max_sent_len/4), 1, 1, activation='relu', border_mode='same')(x)
                five_x_five = Convolution2D(args.max_sent_len, 5, 5, activation='relu', border_mode='same')(five_x_five)
                five_x_five = MaxPooling2D(pool_size=(2, 2), border_mode='same')(five_x_five)

                max_pool = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
                max_pool = Convolution2D(args.max_sent_len, 1, 1, activation='relu', border_mode='same')(max_pool) #TODO: Dim red.

                x = merge([one_x_one, three_x_three, five_x_five, max_pool], mode='concat', name='inception_{0}'.format(inc_idx))


                x = Reshape((args.max_sent_len, args.max_word_len, args.char_embedding_dim))(x)

            feature_size = args.char_embedding_dim//2 * args.max_word_len//2 * 4# + ((args.char_embedding_dim/3) * (maxlen_word/3))
            char_embedding = Reshape((args.max_sent_len, feature_size))(x)

        elif args.resnet:
            prev_x = x
            for _ in range(args.resnet):
                # http://arxiv.org/abs/1603.05027
                # Adapted from github.com/robertostling/sigmorphon2016-system
                if args.bn:
                    x = BatchNormalization(mode=bn_mode)(x)
                x = Activation('relu')(x)
                if args.dropout:
                    x = Dropout(args.dropout)(x)

                x = Convolution2D(args.max_sent_len, 5, 5, activation='relu', border_mode='same')(x)

                # x = Convolution1D(
                #     conv_dims, kernel_size, border_mode='same', init='he_normal',
                #     W_regularizer=conv_regularizer,
                #     b_regularizer=conv_regularizer)(x)

                if args.bn:
                    x = BatchNormalization(mode=bn_mode)(x)
                x = Activation('relu')(x)
                if args.dropout:
                    x = Dropout(args.dropout)(x)

                x = Convolution2D(args.max_sent_len, 5, 5, activation='relu', border_mode='same')(x)

                # x = Convolution1D(
                #     conv_dims, kernel_size, border_mode='same', init='he_normal',
                #     W_regularizer=conv_regularizer,
                #     b_regularizer=conv_regularizer)(x)

                x = merge([prev_x, x], mode='sum')
                prev_x = x

            if args.bn:
                x = BatchNormalization(mode=bn_mode)(x)

            x = Activation('relu')(x)

            feature_size = args.max_word_len * args.char_embedding_dim
            char_embedding = Reshape((args.max_sent_len, feature_size))(x)

        else:

            x = Convolution2D(args.max_sent_len, 4, 8, activation='relu', border_mode='same')(x)
            #
            x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)

            x = Convolution2D(args.max_sent_len, 4, 4, activation='relu', border_mode='same')(x)
            #
            x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
            #avg_p = AveragePooling2D(pool_size=(maxlen_word, 2), border_mode='same')(x)

            #channels = merge([max_p, avg_p], mode='concat')

            feature_size = int(args.max_word_len / 4) * int(args.char_embedding_dim / 4)
            char_embedding = Reshape((args.max_sent_len, int(feature_size)))(x)

    if args.words:
        word_input = Input(shape=(args.max_sent_len, ), dtype='int32', name='word_input')
        if not args.ignore_embeddings:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, dropout=0.1, weights=[embedding_weights], trainable=(args.freeze))(word_input)
        else:
            word_embedding = Embedding(vocab_size, word_embedding_dim, input_length=args.max_sent_len, dropout=0.1)(word_input)

    if args.chars and args.words:
        embedding = merge([word_embedding, char_embedding], mode='concat')
    elif args.words:
        embedding = word_embedding
    elif args.chars:
        embedding = char_embedding

    if args.bn:
        embedding = BatchNormalization(mode=bn_mode)(embedding)

    if args.rnn:
        # Bidirectional GRU
        l = GRU(output_dim=int(args.rnn_dim), return_sequences=True, dropout_W=0.1, dropout_U=0.1, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu', consume_less='cpu')(embedding)
        l = GRU(output_dim=int(args.rnn_dim/2), return_sequences=True, dropout_W=0.1, dropout_U=0.1, activation='relu', consume_less='cpu')(l)

        r = GRU(output_dim=int(args.rnn_dim), return_sequences=True, go_backwards=True, dropout_W=0.1, dropout_U=0.1, input_shape=(args.max_sent_len, word_embedding_dim), activation='relu', consume_less='cpu')(embedding)
        r = GRU(output_dim=int(args.rnn_dim/2), return_sequences=True, go_backwards=True, dropout_W=0.1, dropout_U=0.1, activation='relu', consume_less='cpu')(r)

        x = merge([l, r], mode='concat')
        if args.bn:
            x = BatchNormalization(mode=bn_mode)(x)

    else:
        x = embedding


    # Extra hidden dense
    #x = TimeDistributed(Dense(word_embedding_dim * 4, activation='relu'))(x)

    if args.bypass:
        x = merge([x, bypassed_char_rep], mode='concat')

    x = TimeDistributed(Dense(word_embedding_dim * 2, activation='relu'))(x)

    # Output layer
    main_output = TimeDistributed(Dense(nb_classes, activation='softmax', name='main_output'))(x)

    if args.chars and args.words:
        model_input = [word_input, char_input]
    elif args.chars:
        model_input = [char_input, ]
    elif args.words:
        model_input = [word_input, ]

    if args.aux:
        aux_output = TimeDistributed(Dense(nb_aux_classes, activation='softmax', name='aux_output'))(x)
        model_output = [main_output, aux_output]
    else:
        model_output = [main_output, ]

    if len(args.train) == 2:
        stag_output = TimeDistributed(Dense(nb_stag_classes, activation='softmax', name='stag_output'))(x)
        model_output = [main_output, stag_output]

    model = Model(input=model_input, output=model_output)

    return model

def evaluate(model):
    '''
    TODO: Document
    '''
    print('Dev set results:')


    classes = model.predict(X_dev, batch_size=args.bsize)
    if len(args.train) == 2:
        print('pos')

        dev_classes, dev_accuracy, dev_tags = calculate_accuracy(model, y_dev[0], classes[0])

        print('stag')
        calculate_accuracy(model, y_dev[1], classes[1])
    else:
        dev_classes, dev_accuracy, dev_tags = calculate_accuracy(model, y_dev, classes)

    if args.bookkeeping:
        with open(experiment_dir+'/dev_acc.txt', 'w') as out_f:
            out_f.write('Dev accuracy: {0}\n'.format(dev_accuracy*100))

        save_outputs(dev_tags, X_dev_words, os.path.basename(args.dev[0]).rstrip('.conllu'))

    print('Sanity check on train set:')
    classes = model.predict(X_train, batch_size=args.bsize)
    if len(args.train) == 2:
        print('pos')
        calculate_accuracy(model, y_train[0], classes[0])

        print('stag')
        calculate_accuracy(model, y_train[1], classes[1])
    else:
        calculate_accuracy(model, y_train, classes)

    if args.test:
        print('Test set')
        classes = model.predict(X_test, batch_size=args.bsize)
        if len(args.train) == 2:
            print('pos')
            test_classes, test_accuracy, test_tags = calculate_accuracy(model, y_test[0], classes[0])

            print('stag')
            calculate_accuracy(model, y_test[1], classes[1])

        else:
            test_classes, test_accuracy, test_tags = calculate_accuracy(model, y_test, classes)

        if args.bookkeeping:
            with open(experiment_dir+'/test_acc.txt', 'w') as out_f:
                out_f.write('Test accuracy: {0}\n'.format(test_accuracy*100))

            save_outputs(test_tags, X_test_words, os.path.basename(args.test[0]).rstrip('.conllu'))

    return dev_classes

def calculate_accuracy(model, y, classes):
    '''
    TODO: Document
    '''
    if args.aux:
        classes = classes[0]
        y = y[0]

    sent_tags = []
    corr, err = 0, 0
    for idx, sentence in enumerate(y):
        sent_tags.append([])
        for idy, word in enumerate(sentence):
            gold_tag = np.argmax(word)
            if gold_tag <= 1:
                continue

            pred_tag = np.argmax(classes[idx, idy])
            if pred_tag == gold_tag:
                corr += 1
            else:
                err += 1

            indices = [idx, idy]

            sent_tags[-1].append((indices, gold_tag, pred_tag))

    print('Corr: {0}, Err: {1}'.format(corr, err))
    accuracy = corr / float(corr+err)
    print('Accuracy without dummy labels', accuracy)

    return classes, accuracy, sent_tags


def save_outputs(tags, X_words, fname):
    with open(experiment_dir+'/{0}_tag2id.txt'.format(fname), 'r') as in_f:
        id_to_tag = dict((line.strip().split()[::-1] for line in in_f))

    id_to_word = {}
    with open(experiment_dir+'/{0}_word2id.txt'.format(fname), 'r', encoding='utf-8') as in_f:
        for line in in_f:
            val, key = line.strip().split('\t')
            id_to_word[key] = val


    with open(experiment_dir+'/{0}_outputs.txt'.format(fname), 'w', encoding='utf-8') as out_f:
        for sentence in tags:
            for indices, gold_tag, pred_tag in sentence:
                out_f.write(u'{0}\t{1}\t{2}\n'.format(id_to_word[str(X_words[indices[0], indices[1]])], id_to_tag[str(gold_tag)], id_to_tag[str(pred_tag)]))
            out_f.write('\n')


def make_weight_matrix(X_train):
    '''
    Create sample weights
    '''
    X_weights = np.zeros_like(X_train, dtype=np.float32)
    for idx, sentence in enumerate(X_train):
        for idy, word in enumerate(sentence):
            curr_class = np.argmax(y_train[idx, idy])
            if curr_class == 0:
                X_weights[idx, idy] = 1#e-8
            elif curr_class <= 1:
                X_weights[idx, idy] = 1#e-4
            else:
                X_weights[idx, idy] = 1#0

    return X_weights

def convert_to_embedded(X, rev_map):
    '''
    TODO: Document
    '''
    rev_map = dict([(val, key) for key, val in index_dict.items()])
    X_emb = np.zeros((X.shape[0], X_train.shape[1], word_embedding_dim), dtype=np.float32)
    for idx, sentence in enumerate(X):
        for idy, word in enumerate(sentence):
            if word in rev_map:
                X_emb[idx, idy] = word_vectors[rev_map[word]]

    return X_emb

def summary_writer(string):
    '''
    NOTE: Requires some local changes in Keras.
    TODO: Pull request?
    '''
    with open(experiment_dir+'/model_summary.txt', 'a') as out_f:
        out_f.write(string)
        out_f.write('\n')

def save_run_information():
    '''
    FIXME: Several things not implemented.
    '''
    # NOTE: h5py not currently working on Peregrine

    try:
        from keras.utils.visualize_util import plot
        plot(model, to_file=experiment_dir+'/model.png', show_shapes=True)
    except:
        print('Could not save model plot...')

    try:
        model.save_weights(experiment_dir+'/weights.h5')
    except ImportError:
        print('Could not save weights...')

    json_string = model.to_json()
    with open(experiment_dir+'/architecture.json', 'w') as out_f:
        out_f.write(json_string)

    try:
        model.summary(summary_writer)
    except:
        print('Summary not written')

    try:
        write_confusion_matrix(y_dev, dev_classes, nb_classes)
    except:
        print('Conf matrix not written')

    try:
        prepare_error_analysis(X_dev, y_dev, dev_classes, vocab_size)
    except:
        print('Error analysis not written')

def actual_accuracy(act, pred):
    '''
    Calculate accuracy each batch.
    Keras' standard calculation factors in our padding classes. We don't.
    FIXME: Not always working
    '''
    act_argm  = K.argmax(act, axis=-1)   # Indices of act. classes
    pred_argm = K.argmax(pred, axis=-1)  # Indices of pred. classes

    incorrect = K.cast(K.not_equal(act_argm, pred_argm), dtype='float32')
    correct   = K.cast(K.equal(act_argm, pred_argm), dtype='float32')
    padding   = K.cast(K.equal(K.sum(act), 0), dtype='float32')
    start     = K.cast(K.equal(act_argm, 0), dtype='float32')
    end       = K.cast(K.equal(act_argm, 1), dtype='float32')

    pad_start     = K.maximum(padding, start)
    pad_start_end = K.maximum(pad_start, end) # 1 where pad, start or end

    # Subtract pad_start_end from correct, then check equality to 1
    # E.g.: act: [pad, pad, pad, <s>, tag, tag, tag, </s>]
    #      pred: [pad, tag, pad, <s>, tag, tag, err, </s>]
    #   correct: [1,     0,   1,   1,   1,   1,   0,    1]
    #     p_s_e: [1,     1,   1,   1,,  0,   0,   0,    1]
    #  corr-pse: [0,    -1,   0,   0,   1,   1,   0,    0] # Subtraction
    # actu_corr: [0,     0,   0,   0,   1,   1,   0,    0] # Check equality to 1
    corr_preds   = K.sum(K.cast(K.equal(correct - pad_start_end, 1), dtype='float32'))
    incorr_preds = K.sum(K.cast(K.equal(incorrect - pad_start_end, 1), dtype='float32'))
    total = corr_preds + incorr_preds
    accuracy = corr_preds / total

    return accuracy

if __name__ == '__main__':
    print("use chars?", args.chars)

    if args.embeddings:
        if __debug__: print('Loading embeddings...')
        word_vectors, index_dict, word_embedding_dim = utils.read_word_embeddings(args.embeddings)
        if __debug__: print('Embeddings for {} words loaded'.format(len(word_vectors)))
    else:
        word_embedding_dim = args.word_embedding_dim   ### TODO: if no embeddings given, no index_dict!
        index_dict = {} # HACK: Empty dict will do for now
        word_vectors = None

    if __debug__: print('Loading data...')

    # Word data must be read even if word features aren't used
    (X_train_words, y_train), (X_dev_words, y_dev), (X_test_words, y_test), word_vectors = data_utils.read_word_data(args.train[0], args.dev[0], args.test[0], index_dict, word_vectors, args.max_sent_len)
    nb_classes = y_train.shape[2]

    if len(args.train) == 2:
        (X_train_words_stag, y_train_stag), (X_dev_words_stag, y_dev_stag), (X_test_words_stag, y_test_stag), word_vectors = data_utils.read_word_data(args.train[1], args.dev[1], args.test[1], index_dict, word_vectors, args.max_sent_len)
        nb_stag_classes = y_train_stag.shape[2]

    if args.words:
        if not args.embeddings:
            vocab_size = max(np.max(X_train_words), max(np.max(X_dev_words), np.max(X_test_words)))
        else:
            vocab_size = len(index_dict)
            embedding_weights = np.zeros((vocab_size, word_embedding_dim))
            for word, index in index_dict.items():
                #FIXME: Why are a few words not added?
                if __debug__ and word not in word_vectors:
                    print('word not in vectors', word)
                    continue
                embedding_weights[index,:] = word_vectors[word]

    if __debug__:
        pass
        #print('x_train: ({0: <6}, {1: <3}) y_train: ({2: <6}, {3: <3}, {4: <3})'.format(*X_train_words.shape, *y_train.shape))
        #print('x_dev  : ({0: <6}, {1: <3}) y_dev  : ({2: <6}, {3: <3}, {4: <3})'.format(*X_dev_words.shape, *y_dev.shape))
        #if args.test:
        #    print('x_test : ({0: <6}, {1: <3}), y_test : ({2: <6}, {3: <3})'.format(*X_test_words.shape, *y_test.shape))

    X_train_weights = make_weight_matrix(X_train_words)

    if args.chars:
        if __debug__: print('Loading char features...')

        char_to_id = defaultdict(lambda: len(char_to_id))
        char_to_id['tst']
        for dummy_char in (UNKNOWN, SENT_START, SENT_END, SENT_PAD):
            char_to_id[dummy_char]

        #X_train_chars, X_dev_chars, X_test_chars = read_data(data_utils.read_char_data, char_to_id, None, args.max_sent_len, args.max_word_len)
        X_train_chars, X_dev_chars, X_test_chars = data_utils.read_char_data(args.train[0], args.dev[0], args.test[0], char_to_id, args.max_sent_len, args.max_word_len)
        if len(args.train) == 2:
            X_train_chars_stag, X_dev_chars_stag, X_test_chars_stag = data_utils.read_char_data(args.train[1], args.dev[1], args.test[1], char_to_id, args.max_sent_len, args.max_word_len)

            X_train_chars = np.vstack((X_train_chars, X_train_chars_stag))
            X_dev_chars = np.vstack((X_dev_chars, X_dev_chars_stag))
            X_test_chars = np.vstack((X_test_chars, X_test_chars_stag))

        char_vocab_size = len(char_to_id)
        if __debug__:
            print('{0} char ids'.format(char_vocab_size))
            print('x_train chars: ({0: <6}, {1: <3})'.format(X_train_chars[0].shape[0], X_train_chars[0].shape[1]))
            print('x_dev chars  : ({0: <6}, {1: <3})'.format(X_dev_chars[0].shape[0], X_dev_chars[0].shape[1]))
            if args.test:
                print('x_test chars : ({0: <6}, {1: <3}'.format(X_test_chars[0].shape[0], X_test_chars[0].shape[1]))

    if __debug__: print('Building model...')

    if len(args.train) == 2:
        X_train_words = np.vstack((X_train_words, X_train_words_stag))
        X_dev_words = np.vstack((X_dev_words, X_dev_words_stag))

        y_train_main = np.vstack((y_train, np.zeros((y_train_stag.shape[0], y_train_stag.shape[1], y_train.shape[2]), dtype=np.int32)))
        y_dev_main = np.vstack((y_dev, np.zeros((y_dev_stag.shape[0], y_dev_stag.shape[1], y_dev.shape[2]), dtype=np.int32)))

        y_train_stag = np.vstack((np.zeros((y_train.shape[0], y_train.shape[1], y_train_stag.shape[2]), dtype=np.int32), y_train_stag))
        y_dev_stag = np.vstack((np.zeros((y_dev.shape[0], y_dev.shape[1], y_dev_stag.shape[2]), dtype=np.int32), y_dev_stag))

        if args.test:
            X_test_words = np.vstack((X_test_words, X_test_words_stag))
            y_test_main = np.vstack((y_test, np.zeros((y_test_stag.shape[0], y_test_stag.shape[1], y_test.shape[2]), dtype=np.int32)))
            y_test_stag = np.vstack((np.zeros((y_test.shape[0], y_test.shape[1], y_test_stag.shape[2]), dtype=np.int32), y_test_stag))

    if args.chars and args.words:
        X_train = [X_train_words, X_train_chars]
        X_dev = [X_dev_words, X_dev_chars]
        X_test = [X_test_words, X_test_chars]

    elif args.chars:
        X_train = [X_train_chars, ]
        X_dev = [X_dev_chars, ]
        X_test = [X_test_chars, ]
    elif args.words:
        X_train = [X_train_words, ]
        X_dev = [X_dev_words, ]
        X_test = [X_test_words, ]

    if args.aux:
        y_train_aux = utils.map_y_to_abstract(y_train)
        y_dev_aux = utils.map_y_to_abstract(y_dev)
        y_test_aux = utils.map_y_to_abstract(y_test)
        nb_aux_classes = y_train_aux.shape[2]

        y_train = [y_train, y_train_aux]
        y_dev = [y_dev, y_dev_aux]
        y_test = [y_test, y_test_aux]
        model_losses = ['categorical_crossentropy', 'categorical_crossentropy']
        model_loss_weights = [1.0, 0.5]
        model_metrics = [actual_accuracy, ]

    elif len(args.train) == 2:
        y_train = [y_train_main, y_train_stag]
        y_dev = [y_dev_main, y_dev_stag]
        if args.test:
            y_test = [y_test_main, y_test_stag]
        model_losses = ['categorical_crossentropy', 'categorical_crossentropy']
        model_loss_weights = [1.0, 0.1]
        model_metrics = ['accuracy', actual_accuracy, ]

    else:
        model_outputs = [y_train, ]
        model_losses = ['categorical_crossentropy', ]
        model_loss_weights = [1.0, ]
        model_metrics = [actual_accuracy, ]


    model = build_model()

    model.compile(optimizer='adam',
              loss=model_losses,
              loss_weights=model_loss_weights,
              metrics=model_metrics)

    model.summary()

    if __debug__: print('Fitting...')

    callbacks = [ProgbarLogger()]

    if args.bookkeeping:
        if 'tensorflow' in K._BACKEND:
            callbacks.append(TensorBoard(log_dir=experiment_dir, histogram_freq=1, write_graph=False)) ##write graph doesn't work on my laptop: unexpected keyword argument 'write_graph' (I've tensorflow 0.90rc, on peregrine we have 0.80)
        if args.model_checkpoint:
            try:
                import h5py
                weights_fname = experiment_dir+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
                callbacks.append(ModelCheckpoint(filepath=weights_fname, verbose=0, save_best_only=True))
            except ImportError:
                print('Can not save models (no h5py)')

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))



    if args.memsave:
        n_batches = len(X_train[0]) // args.bsize
        n_dev_batches = len(X_dev[0]) // args.bsize
        for epoch in range(args.epochs):
            print('Epoch {0}'.format(epoch))
            for batch in range(n_batches):
                curr_batch_idx = np.arange(args.bsize * batch, args.bsize * (batch+1))
                model.train_on_batch([X_train[0][curr_batch_idx], ], y_train[curr_batch_idx])

            print('validating')
            dev_loss = [0, 0, 0]
            for batch in range(n_dev_batches):
                curr_batch_idx = np.arange(args.bsize * batch, args.bsize * (batch+1))
                loss = model.test_on_batch([X_dev[0][curr_batch_idx], ], y_dev[curr_batch_idx])

                for idx, num in enumerate(loss):
                    dev_loss[idx] += num

            dev_loss = np.asarray(dev_loss, dtype=np.float32)
            dev_loss /= n_dev_batches
            print('loss: {0}, auto-acc: {1}, real acc: {2}'.format(*dev_loss))

    else:
        model.fit(X_train, y_train,
                      validation_data=(X_dev, y_dev),
                      nb_epoch=args.epochs,
                      batch_size=args.bsize,
                      callbacks=callbacks,
                      verbose=args.verbose)

    if __debug__:
        print(args)
        print('Evaluating...')

    dev_classes = evaluate(model)

    if args.bookkeeping:
        save_run_information()

    print('Completed: {0}'.format(experiment_tag))
