from features.vocab import Vocab


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


class FeatureFunction():

    def __init__(self):
        pass

    def inform(self,data):
        return self.process(data)

    def lookup(self,data):
        return self.process(data)

    def process(self,data):
        pass


class LexFeatureFunction(FeatureFunction):

    def __init__(self):
        super().__init__()
        self.vocab = Vocab()

    def inform(self,data):
        generated = self.process(data)
        self.vocab.add(generated)
        self.vocab.generate_dict()

    def lookup(self,data):
        size = len(data)
        processed = self.process(data)

        lu = self.vocab.lookup_sparse(processed,size)
        return lu





class UnigramFeatureFunction(LexFeatureFunction):
    def process(self,data):
        return map(lambda text: [w for w in " ".join(text.split()).split()], data)

class BigramFeatureFunction(LexFeatureFunction):
    def process(self, data):
        return  map(lambda text: ["_".join(ng) for ng in ngrams(" ".join(text.split()), 2)], data)

class CharNGramFeatureFunction(LexFeatureFunction):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def process(self,data):
        return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), self.size)], data)




