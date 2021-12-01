"""
Name         : hw3.py
Author       : Liam Richards (lrm444)
Version      : 2.0
Date Created : November 7th, 2021
Description  : This is a simple stochastic Part-of-Speech tagger.

To score the program:
py -2 score.py WSJ_24.pos submission.pos

"""

import math
from tqdm import tqdm     # for a progress bar

## Training Corpus & Test Corpus##
train_corpus = open("WSJ_02-21.pos")
train_list = train_corpus.readlines()
test_corpus = open("WSJ_23.words")
test_list = test_corpus.readlines()

def prepareData():
    '''
    Processing initial training corpus data
    - compile list of all POS tags, including begin/end of sent
    - get freq of vocab
    - vocab with freq < 2 is regarded as UNKNOWN
    - return list of unique vocab and POS tags
    '''

    vocab = dict()  # {word: freq}
    POS_tags = list()

    for i in range(len(train_list)):
        # Note: split("\t") results in "\n" at end, split() automatically strips the line
        split_line = train_list[i].split()
        if len(split_line) != 0:
            word = split_line[0]
            tag = split_line[1]
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
            if tag not in POS_tags:
                POS_tags.append(tag)

    vocab_list = []
    for word in vocab.keys():
        if vocab[word] >= 2:
            vocab_list.append(word)
    vocab_list.append("UNKNOWN")
    vocab_list.append("SOS")
    vocab_list.append("EOS")

    POS_tags.append("Begin_Sent")
    POS_tags.append("End_Sent")

    return vocab_list, POS_tags

### Calculate the transmission and emission probabilities ###
def calc_probabilities(vocab_list, tags):
    '''
    Construct a Hidden Markov Model from the training corpus
    - get freq of words emitting given a POS tag
    - keep track of next split line to record freq of transitions between POS tags
    - convert freq of emission and transition to probabilities
    - return emission and transition dicts
    '''
    vocab_set = set(vocab_list)     # uses hashing, much faster lookup time
    emit_p = dict()     # {tag: {word: freq/prob}}
    trans_p = dict()    # {tag: {tag: freq/prob}}

    print("Getting freq of words...")
    for i in tqdm(range(len(train_list))):
        # skip blank lines
        split_line = train_list[i].split()
        if len(split_line) != 0:
            word = split_line[0]
            tag = split_line[1]

            # Unknown words, freq < 2
            if word not in vocab_set:
                word = "UNKNOWN"

            # Update emission freq for word given tag
            if tag in emit_p:
                if word in emit_p[tag]:
                    emit_p[tag][word] += 1
                else:
                    emit_p[tag][word] = 1
            else:
                emit_p[tag] = {word: 1}

            # Update transition freq between tags
            # start of corpus
            if i == 0:
                trans_p["Begin_Sent"] = {tag: 1}
                emit_p["Begin_Sent"] = {"SOS": 1}

            # end of corpus
            elif i >= len(train_list) - 2:
                trans_p[tag] = {"End_Sent": 1}
                emit_p["End_Sent"] = {"EOS": 1}
            
            # else
            else:
                if train_list[i+1] == "\n":
                    next_split_line = train_list[i+2].split()
                    next_tag = next_split_line[1]
                else:
                    next_split_line = train_list[i+1].split()
                    next_tag = next_split_line[1]

                if tag in trans_p:
                    if next_tag in trans_p[tag]:
                        trans_p[tag][next_tag] += 1
                    else:
                        trans_p[tag][next_tag] = 1
                else:
                    trans_p[tag] = {next_tag: 1}

    ## convert freq to probabilities ##
    # emissions
    # prob(word | tag) = freq of word / total freq of tag
    for tag in tags:
        total = 0
        for word in emit_p[tag]:
            total += emit_p[tag][word]

        for word in vocab_list:
            freq = 0
            if word in emit_p[tag]:
                freq = emit_p[tag][word]

            # emission dict now has probability for ALL words given tag
            # smoothing will make or break the program, cannot simply divide freq / total
            emit_p[tag][word] = (freq + 0.001) / (total + 0.001 * len(vocab_list))  # smoothing
    
    # transitions
    # prob(next_tag | tag) = freq of next_tag / total freq of tag
    for tag in tags:
        total = 0
        for next_tag in tags:
            if tag in trans_p and next_tag in trans_p[tag]:
                total += trans_p[tag][next_tag]
        
        for next_tag in tags:
            freq = 0
            if tag in trans_p:
                if next_tag in trans_p[tag]:
                    freq = trans_p[tag][next_tag]
                # smoothing necessary to have a non-zero value for transition probabilites
                trans_p[tag][next_tag] = (freq + 0.001) / (total + 0.001 * len(vocab_list))  # smoothing
            else:
                # this control path is needed to store prob for End_Sent
                trans_p[tag] = {next_tag: (freq + 0.001) / (total + 0.001 * len(vocab_list))}  # smoothing
    
    return emit_p, trans_p

# Viterbi algorithm structure adapted from https://en.wikipedia.org/wiki/Viterbi_algorithm
def viterbi(observ, emit_table, trans_p):
    '''
    Executing the Viterbi Algorithm
    '''
    max_probs = [[0] * len(observ) for i in range(len(tags))]
    max_tags = [[None] * len(observ) for i in range(len(tags))]

    max_score = 0
    max_i = None

    print("Executing Viterbi Algorithm:")

    #intitialize the starting word of sentence
    for row in tqdm(range(len(tags))):
        s = tags[row]
        if s in trans_p["Begin_Sent"]:
            score = math.log(emit_table[row][0]) + math.log(trans_p["Begin_Sent"][s])
        else:
            score = 0
        max_probs[row][0] = score
        max_tags[row][0] = 0

    for col in tqdm(range(1, len(observ))):    
        for t in range(len(tags)):  
            max_score = float("-inf")
            max_i = None
            for pt in range(len(tags)):

                score = max_probs[pt][col-1] + math.log(trans_p[tags[pt]][tags[t]]) + math.log(emit_table[t][col])

                if score > max_score:
                    max_score = score
                    max_i = pt

            max_probs[t][col] = max_score
            max_tags[t][col] = max_i
        
    return max_probs, max_tags

def predictTags(tags, trans_p, emit_p, vocab_list): 
    '''
    Predicting the tags for the test corpus
    - compile list of 'observed' words (in order of observation)
    - construct an emission table for observed words
    - run viterbi algorithm, getting 
    - return optimal list of POS tags
    '''
    vocab_set = set(vocab_list) # set uses hashing, much faster lookup time
    observ = [] 
    observ.append("SOS")

    for i in range(len(test_list)):
        # skip blank lines
        if test_list[i] != "\n":
            word = test_list[i].rstrip()

            if word not in vocab_set:
                word = "UNKNOWN"
            observ.append(word)

    observ.append("EOS")

    ##Emission likelihood table for observ, i.e. a simple transducer
    # cells contain the likelihood a particular word is a given tag 
    # at a given location in a sentence
    # rows: POS_tags, cols: emission prob of all observed words in order
    emit_table = []
    for tag in tags:
        cols = []
        for word in observ:
            cols.append(emit_p[tag][word])
        emit_table.append(cols)
    
    # Viterbi
    max_probs, max_tags = viterbi(observ, emit_table, trans_p)

    ##Choose the highest POS tags##
    best_idx = [None] * len(observ)
    predicted_tags = [None] * len(observ)

    # Get the most probable state and its backtrack
    argmax = max_probs[0][len(observ) - 2]
    best_idx[len(observ) - 2] = 0
    for t in range(1, len(tags)):
        if max_probs[t][len(observ) - 2] > argmax:
            argmax = max_probs[t][len(observ) - 2]
            best_idx[len(observ) - 2] = t

    predicted_tags[len(observ) - 2] = tags[best_idx[len(observ) - 2]]

    # Follow the backtrack until the first observations
    for i in range(len(observ) - 2, 1, -1):
        best_idx[i - 1] = max_tags[best_idx[i]][i]
        predicted_tags[i - 1] = tags[best_idx[i - 1]]
        
    return predicted_tags

def writeToFile(predicted_tags):
    '''
    writing predicted tags to file, excluding SOS and EOS tags
    '''
    f = open("submission.pos", "w")

    # exclude start and end tags (SOS and EOS)
    predicted_tags = predicted_tags[1:-1]
    pred_ctr = 0
    for i in range(len(test_list)):
        word = test_list[i].rstrip()
        if test_list[i] == '' or test_list[i] == "\n":
            f.write('\n')
        else:
            f.write(word + "\t" + predicted_tags[pred_ctr] + "\n")
            pred_ctr += 1

    f.close()

if __name__ == '__main__':
    vocab_list, tags = prepareData()
    emit_p, trans_p = calc_probabilities(vocab_list, tags)
    predicted_tags = predictTags(tags, trans_p, emit_p, vocab_list)
    writeToFile(predicted_tags)