"""
Sentiment Analysis is very popular in the areas of research and social media. It is a sort of Opinion Mining.
The results might not be very accurate, but can still be useful.

NLTK modules is popular in python which can be used for classifying the text in the given input text.
The categories are:
1> Good
2> Bad

Using a boolean word feature extraction as a part of the simple Naive Bayes Classifies as baseline, we can start.


Bag of Words Feat

"""

import nltk
import random
from nltk.corpus import movie_reviews

# Standardizing the list of all the words in the corpus
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# Creating a frequency distribution of the words
all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))
# This also contains punctuations. Which is not useful. Lets take of those words later.
""" [(',', 77717), ('the', 76529), ('.', 65876), ('a', 38106), ('and', 35576), ('of', 34123), ('to', 31937), ("'", 30585), ('is', 25195), ('in', 21822), ('s', 18513), ('"', 17612), ('it', 16107), ('that', 15924), ('-', 15595)]
"""
print(all_words['bad'])   # 253 movie reviews have 'bad' in them

word_features = list(all_words.keys())[:3000]

#
# list of (tuples of (list of words) with their categories))
# This will be the used for training and testing the classifier
documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
for fileid in movie_reviews.fileids('neg'):
    print(fileid)
