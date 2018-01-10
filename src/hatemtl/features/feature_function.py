from common.features.feature_function import FeatureFunction
from hatemtl.features.vocab import Vocab
import numpy as np

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def char_ngrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


class LexFeatureFunction(FeatureFunction):

    def __init__(self,max_size=5000,naming=""):
        super().__init__()
        self.vocab = Vocab(max_size)
        self.naming = naming

    def inform(self,*datasets):
        print("Inform features")
        for dataset in datasets:
            if dataset is not None:
                generated = self.process(dataset)
                self.vocab.add(generated)
        print("Gen dict")
        self.vocab.generate_dict()

    def lookup(self,data):
        size = len(data)
        processed = self.process(data)

        lu = self.vocab.lookup_sparse(processed,size)
        return lu



class UnigramFeatureFunction(LexFeatureFunction):
    def process(self,data):
        return map(lambda item: [w for w in " ".join(item["data"].split()).split()], data)

    def get_name(self):
        return self.naming + "-" + UnigramFeatureFunction.__name__

class BigramFeatureFunction(LexFeatureFunction):
    def process(self, data):
        return  map(lambda item: ["_".join(ng) for ng in ngrams(" ".join(item["data"].split()), 2)], data)

    def get_name(self):
        return self.naming + "-" + BigramFeatureFunction.__name__

class CharNGramFeatureFunction(LexFeatureFunction):
    def __init__(self, size, naming=""):
        super().__init__(naming=naming)
        self.size = size

    def process(self,data):
        return map(lambda item: ["".join(ng) for ng in char_ngrams(" ".join(item["data"].split()), self.size)], data)

    def get_name(self):
        return self.naming + "-" + CharNGramFeatureFunction.__name__+"-"+str(self.size)


class EmbeddingFeatureFunction(FeatureFunction):
    def __init__(self, embedding_file, preprocessors, separator=None,
                 naming=""):
        super().__init__()
        self.preprocessors = preprocessors
        self.embedding_file = embedding_file
        self.embeddings = self.read_embeddings(self.embedding_file,
                                               separator=separator)
        self.OOV = self.average_embedding(self.embeddings)
        self.naming = naming

    @staticmethod
    def read_embeddings(embeddings_file, separator=None):
        w2v = {}
        for line in open(embeddings_file):
            w, vec = line.strip().split(separator, 1)
            w2v[w] = np.array([float(x) for x in vec.split(separator)])
        return w2v

    @staticmethod
    def average_embedding(embeddings):
        return np.average(np.stack([v for v in embeddings.values()]), axis=0)

    def process(self, data):
        for processor in self.preprocessors:
            data = processor.transform_batch(data)
        return np.stack(map(lambda item: np.average([self.embeddings.get(w, self.OOV) for w in item["data"].split()],axis=0), data),axis=0)

    def inform(self,*datasets):
        pass

    def get_name(self):
        return self.naming + "-" + EmbeddingFeatureFunction.__name__+"-"+str(
            self.embedding_file)
