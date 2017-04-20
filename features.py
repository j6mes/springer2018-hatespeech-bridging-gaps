import re
from pprint import pprint
import json

from dataset_reader import DataSet
from preprocessing import preprocess


def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output


def char_ngrams(input, n):
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output


def feature_word_unigrams(text):
    return " ".join(text.split()).split()


def feature_word_bigrams(text):
    return ["_".join(ng) for ng in ngrams(" ".join(text.split()),2)]


def feature_char_unigrams(text):
    return ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 1)]


def feature_char_bigrams(text):
    return ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 2)]


def feature_char_trigrams(text):
    return ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 3)]





def features(text,tools=list([feature_word_unigrams,
                             feature_char_unigrams,
                             feature_char_bigrams,
                             feature_char_trigrams])):
    feats = []
    for tool in tools:
        feats.extend(tool(text))

    return feats



if __name__=="__main__":
    racism = DataSet("Racism")


    from collections import defaultdict
    all_features = defaultdict(int)
    for tweet in racism.data:
        print(tweet['text'])
        print(features(preprocess(tweet)))


