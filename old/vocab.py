from collections import defaultdict

from tqdm import tqdm

from dataset_reader import DataSet
from features import features
from preprocessing import preprocess


class Vocab():
    def __init__(self):
        self.vocab = set()
        self.vocab.add("UNKNOWN")

    def add(self,all_items):
        for item in tqdm(all_items):
            for f in item:
                self.vocab.add(f)

    def generate_dict(self):
        vocab = dict()
        for i,word in enumerate(self.vocab):
            vocab[word] = i
        self.vocab = vocab

    def lookup(self,instance):
        ret = defaultdict(int)
        for feature in instance:
            if feature in self.vocab:
                ret[self.vocab[feature]] += 1
            else:
                ret[self.vocab["UNKNOWN"]] += 1
        return ret


if __name__ == "__main__":
    racism = DataSet("racism")
    racism_features = []
    for tweet in tqdm(racism.data):
        racism_features.append(features(preprocess(tweet)))

    sexism = DataSet("sexism")
    sexism_features = []
    for tweet in tqdm(sexism.data):
        sexism_features.append(features(preprocess(tweet)))

    neither = DataSet("neither")
    neither_features = []
    for tweet in tqdm(neither.data):
        neither_features.append(features(preprocess(tweet)))

    vocab = Vocab()
    vocab.add(racism_features)
    vocab.add(sexism_features)
    vocab.add(neither_features)

