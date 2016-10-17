#!/usr/bin/env python

import os
import re
import codecs
import numpy as np
from collections import defaultdict
from config import *

from codecs import open

def read_word_embeddings(fname):
    word_vec_map = {}
    word_id_map = {}
    with open(fname, 'r', encoding='utf-8') as in_f:
        for idx, line in enumerate(in_f):
            fields = line.strip().split()
            word = fields[0]
            embedding = np.asarray([float(i) for i in fields[1:]], dtype=np.float32)

            word_vec_map[word] = embedding
            word_id_map[word] = len(word_id_map)

    # get dimensions from last word added
    vec_dim = len(word_vec_map[word])
    return word_vec_map, word_id_map, vec_dim

def load_character_data(fname, char_to_id, max_sent_len, max_word_len=32):
    X = []

    ## first load all (except cut on word length
    with open(fname, 'r', encoding='utf-8') as in_f:
        curr_X = []
        for line in in_f:
            line = line.strip()
            if not line:
                if curr_X:
                    X.append(curr_X)
                    curr_X = []
                continue

            token = line.split('\t')[0]
            #if len(token) > max_word_len:
            #    char_ids = [char_to_id[UNKNOWN]] #todo: keep <max info
            #else:
            char_ids = [char_to_id[char] for char in token]
            curr_X.append(char_ids)

            if args.shorten_sents and len(curr_X) >= max_sent_len-2:
                X.append(curr_X)
                #y.append(curr_y)
                curr_X = []#[char_to_id[SENT_CONT]]]#word_to_id[SENT_CONT]]
                #curr_y = [tag_to_id[SENT_CONT]]

    # Final sent
    if curr_X:
        X.append(curr_X)

    # cutoff for max_sent_len
    X = [s[:max_sent_len] for s in X]
    return X

def load_word_data(fname, word_to_id, tag_to_id, max_sent_len, is_training=False):
    """
    Reading of CoNLL file, converting to word ids.
    If is_training, unknown mwes will be added to the embedding dictionary using a heuristic.
    """
    print("reading ",fname)
    X, y = [], []
    splittable = set((',', '.', '"', "'", '-', ':'))

    ## first read in all
    with open(fname, 'r', encoding='utf-8') as in_f:
        curr_X, curr_y = [], []
        for line in in_f:
            line = line.strip()
            if not line:
                # Make sure no lists are empty
                if curr_X and curr_y:
                    X.append(curr_X)
                    y.append(curr_y)

                    curr_X = []
                    curr_y = []

                continue

            token, tag = line.strip().split('\t')

            #if args.multilingual:
            #    token = langcode + ':' + token

            # Some preprocessing
            if token not in splittable and re.match('^[0-9\.\,-]+$', token):
                curr_X.append(word_to_id[NUMBER])
            elif token in word_to_id:
                curr_X.append(word_to_id[token])
            elif token.lower() in word_to_id:
                curr_X.append(word_to_id[token.lower()])
            elif token.upper() in word_to_id:
                curr_X.append(word_to_id[token.upper()])
            elif token.capitalize() in word_to_id:
                curr_X.append(word_to_id[token.capitalize()])
            elif is_training and args.mwe and ('~' in token or '-' in token):
                curr_X.append(attempt_reconstruction(token, word_to_id))
            else:
                #print("unk*****", token) #if token not in embeddings it's UNK (or mwu if option off)
                curr_X.append(word_to_id[UNKNOWN])
            curr_y.append(tag_to_id[tag])

            if args.shorten_sents and len(curr_X) >= max_sent_len-2:
                X.append(curr_X)
                y.append(curr_y)
                curr_X = []#word_to_id[SENT_CONT]]
                curr_y = []#tag_to_id[SENT_CONT]]

    # Final sent
    if curr_X and curr_y:
        X.append(curr_X)
        y.append(curr_y)

    ## get some stats on dataset
    sent_lens = [len(s) for s in X]
    max_sent_len_data = max(sent_lens)
    percentile = int(np.percentile(sent_lens, 90))
    ## max sentence cutoff
    # Two options: either discard sentences (as in earlier code), or use all up to max_sent_len (
    # Leave room for padding
    old_len = len(X)
    discarded_tokens = 0
    for idx, s in enumerate(X):
        if len(s) > max_sent_len-2:
            discarded_tokens += len(s) - (max_sent_len - 2)

    X = [s[:max_sent_len-2] for s in X]
    y = [s[:max_sent_len-2] for s in y]

    if __debug__:
        print('max len in dataset: {0}\t90-percentile: {2}\tmax len used: {1}'.format(max_sent_len_data, max_sent_len, percentile))
        print('n discarded sents: {0}'.format(old_len - len(X)))
        print('n discarded toks: {0}'.format(discarded_tokens))

    if args.bookkeeping:
        dsetname = os.path.basename(fname).rstrip('.conllu')
        save_ids(word_to_id, tag_to_id, dsetname)

    return X, y, word_to_id, tag_to_id

def attempt_reconstruction(complex_token, word_to_id):
    constituents = re.split('[~-]+', complex_token)
    token_ids = [word_to_id[token] for token in constituents if token in word_to_id]

    if len(token_ids) == 0:
        return word_to_id[UNKNOWN]
    else:
        word_to_id[complex_token] = len(word_to_id)
        token_ids.append(word_to_id[complex_token])
        return token_ids

def save_ids(word_to_id, tag_to_id, fname):
    write_mapping(word_to_id, experiment_dir+'/{0}_word2id.txt'.format(fname))
    write_mapping(tag_to_id, experiment_dir+'/{0}_tag2id.txt'.format(fname))

def map_y_to_abstract(y):
    tag_to_id = {}
    fname = os.path.basename(args.train[0]).rstrip('.conllu')
    with codecs.open(experiment_dir+'/{0}_tag2id.txt'.format(fname), 'r', encoding='utf-8') as in_f:
        for line in in_f:
            key, value = line.strip().split('\t')
            tag_to_id[key] = int(value)

    y_aux = np.zeros((y.shape[0], y.shape[1], len(abstract_tagmap)), dtype=np.int32)
    for idx, line in enumerate(y):
        for idy, one_hot_y in enumerate(line):
            if np.sum(one_hot_y) == 0:
                continue

            tag_id = np.argmax(one_hot_y)
            for semtag, stored_id in tag_to_id.items():
                if stored_id == tag_id:
                    current_tag = semtag
                    break

            for abstract_tag, semtags in abstract_tagmap.items():
                if current_tag in semtags:
                    abstract_id = abstract_tag_to_id[abstract_tag]
                    y_aux[idx, idy, abstract_id] = 1
                    break

    return y_aux

def write_mapping(tok_tag_map, fname):
    with codecs.open(fname, 'w', encoding='utf-8') as out_f:
        for key, value in tok_tag_map.items():
            out_f.write(u'{0}\t{1}\n'.format(key, value))
