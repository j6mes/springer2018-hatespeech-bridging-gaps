from common.features.feature_function import FeatureFunction
from hatemtl.features.vocab import Vocab

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


