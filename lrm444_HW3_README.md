# Part of Speech Tagger
In this assignment, I am constructing a stochastic part of speech tagger (POS tagger), which assigns each word in a text a part-of-speech label. As opposed to a rule-based tagger, a stochastic tagger uses a training corpus to compute the probability of a given word's tag in a given context. I will be following Hidden Markov Model approach using the Viterbi algorithm.

Training corpus: ```WSJ_02-21.pos``` (950K words)
Development corpus: ```WSJ_24.pos``` (32.9K words)
Test corpus: ```WSJ_23.words``` (56.7K words)

When I ran this POS tagger against the development corpus, I obtained a 94.5% accuracy score.

### :rocket: Running the program
Run with the following command:
```python HW3.py```
This should output a file titled *submission.pos*, which contains the original file words alongside their respective predicted part-of-speech tags.

### Unknown Words (OOV)
To be complete, my POS tagger needs a method to handle previously unknown words, such as acronyms, proper names and words that don't exist in the training corpus. The strategy I used to handle the "likelihood" of an OOV was by treating all unknown words as a single word with the tag *UNKNOWN*.

To elaborate, I used the distribution of words in the training corpus that occur only once as a basis for the likelihood of OOV items. For instance, if there are 50K *UNKNOWN* words that only occur once in a corpus of size 950K, the likelihood a given word is *UNKNOWN* is 50K / 950K. 