# Create POS Dictionary
from typing import Dict


pos_dict = {}
state = {}

# Convert to probabilities
# ================================================
def convert_to_probabilities(dict): 
    # Get total counts for each POS
    totals_dict = {}

    # For each key (DT), add up all the values of the (DT) dictionary
    for key, value in dict.items():
        total = 0
        for value in dict[key].values():
            total = total + value   # key: IN, total: 98554
        
        # Store POS / POS Total count pair in dict
        totals_dict.update({key: total})

    # Convert to probabilities for each word GIVEN the POS
    # Ex: Given DT, probability of 'the' is (freq of 'the') / (total freq of DT)
    for key, value in dict.items():
        for innerkey in dict[key].keys():
            dict[key][innerkey] = dict[key][innerkey] / totals_dict[key]    #overwriting existing dictionary
            # print(key, dict[key][innerkey])


# Viterbi algorithm structure adapted from https://en.wikipedia.org/wiki/Viterbi_algorithm
def viterbi(pos, words, start_p, trans_p, emit_p):
    V = [{}]
    for w in words:
        V[0][w] = {"prob": start_p[w] * emit_p[w][pos[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(pos)):
        V.append({})
        for w in words:
            max_tr_prob = V[t - 1][words[0]]["prob"] * trans_p[words[0]][w]
            prev_st_selected = words[0]
            for prev_st in words[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][w]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[w][pos[t]]
            V[t][w] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for w, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = w
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ("The steps of words are " + " ".join(opt) + " with highest probability of %s" % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " " * 5 + "     ".join(("%3d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%lf" % v[state]["prob"]) for v in V)


# Run the WSJ_24 pos_dict frequencies file
with open('WSJ_02-21.pos') as f:

    # The very first state is a beginning of the sentence
    prev_word = "\n"
    prev_tag = "Begin_End_Sent"

    # for each word/tag pair in file
    for line in f:
        line = line.strip()

        # Skip over blank lines in file for POS tagging
        if(line != ""):
            curr_word = line.split('\t')[0]  
            curr_tag = line.split('\t')[1]

            # Create dictionary inside POS dictionary (if DNE)
            if(curr_tag not in pos_dict.keys()):
                pos_dict[curr_tag] = {}   # {'DT': {...}}

            # Increment the freq count of curr_word if already exists
            if(curr_word in pos_dict[curr_tag].keys()):
                pos_dict[curr_tag][curr_word]+= 1  # {'DT': {'The': 214}}
            # If not found, freq count of curr_word starts at 1
            else:
                pos_dict[curr_tag][curr_word] = 1  # {'DT': {'All': 1}}
    

        # Table of frequencies of the following states
        # ========================================================
        # state [prev_tag] = {'current tag': freq}
        if(line == "" or line=="\n"):
            curr_word = ""
            curr_tag = "Begin_End_Sent"

        # Create dictionary inside state dictionary (if DNE)
        if(prev_tag not in state.keys()):
            state[prev_tag] = {}

        # Increment the freq count of curr_word if already exists
        if(curr_tag in state[prev_tag].keys()):
            state[prev_tag][curr_tag] += 1
        # If not found, freq count of curr_word starts at 1
        else:
            state[prev_tag][curr_tag] = 1
            
        prev_word = curr_word
        prev_tag = curr_tag
    

    convert_to_probabilities(pos_dict)
    convert_to_probabilities(state)
        
    print(pos_dict)
    print(state)

    # Run the Viterbi algorithm on the two probability dictionaries to determine the best 
    



# UNNECESSARY CODE???

with open("WSJ_02-21.pos") as f:
    # total # of words is length of file
    data = f.read()
    words = data.split()
    total = len(words)
            

# Recursively iterate through nested dictionary values (borrowed from https://thispointer.com/python-how-to-iterate-over-nested-dictionary-dict-of-dicts/)
def iterate_nested_dict(dict_obj):
    ''' This function accepts a nested dictionary as argument
        and iterate over all values of nested dictionaries
    '''
    total = 0
    # Iterate over all key-value pairs of dict argument
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in iterate_nested_dict(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value (produces sequence of values rather than 1 big output)
            total = total + value
            print(key, total)
            yield (key, value)

#Loop through all key-value pairs of a nested dictionary
# for pair in iterate_nested_dict(pos_dict):
#     print(pair)
    