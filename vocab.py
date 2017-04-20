from collections import defaultdict

from tqdm import tqdm

from dataset_reader import DataSet
from features import features
from preprocessing import preprocess


class Vocab():
    def __init__(self):
        self.vocab = defaultdict(int)

    def add(self,all_items):
        for item in tqdm(all_items):
            for f in item:
                self.vocab[f] +=1


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

    print(vocab.vocab)