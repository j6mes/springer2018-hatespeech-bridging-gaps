import os

from torch.autograd import Variable
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from dataset.batcher import Batcher
from dataset.data_set import DataSet, CompositeDataset
from dataset.formatter import TextAnnotationFormatter
from dataset.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema
from dataset.reader import CSVReader, JSONLineReader

from features.feature_function import UnigramFeatureFunction, BigramFeatureFunction, CharNGramFeatureFunction
from model.multi_layer import MLP
from scipy.sparse import hstack

def transpose(l):
    return list(map(list, zip(*l)))

def prepare(data,labels):
    data = data.todok()
    i = torch.from_numpy(np.array(transpose(list(data.keys())),dtype=np.int64))
    v = torch.FloatTensor(list(data.values()))
    return Variable(torch.sparse.FloatTensor(i, v, torch.Size(data.shape))), Variable(torch.LongTensor(labels))



def prepare2(data,labels):
    data = data.todense()
    v = torch.FloatTensor(np.array(data))
    return Variable(v), Variable(torch.LongTensor(labels))


def train(model,data,batch_size,lr,epochs):

    class Simple(nn.Module):
        def __init__(self,inf,outf):
            super().__init__()
            self.fc = nn.Linear(inf,outf)

        def forward(self, d):
            return self.fc(d)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = 0

    batcher = Batcher(data,batch_size)



    for epoch in range(epochs):
        for batch,size in batcher:
            d,labels = prepare2(*batch)

            optimizer.zero_grad()
            logits = model(d)

            loss = F.cross_entropy(logits,labels)

            loss.backward()
            optimizer.step()





class Features():
    def __init__(self,features=list(),preprocessing=None):
        self.preprocessing = preprocessing
        self.feature_functions = features
        self.vocabs = dict()

    def load(self,dataset):
        fs = []
        preprocessed = self.preprocess_all(dataset.data)
        for feature_function in self.feature_functions:
            print("Load {0}".format(feature_function))
            fs.append(feature_function.lookup(preprocessed))
        return hstack(fs),self.labels(dataset.data)

    def labels(self,data):
        return [datum["label"] for datum in data]

    def preprocess_all(self,data):
        return list(
            map(
                lambda datum: self.preprocessing(datum["data"]) if self.preprocessing is not None else datum["data"],
                data))

    def generate_vocab(self,dataset):
        preprocessed = self.preprocess_all(dataset.data)
        print(len(preprocessed))
        for feature_function in self.feature_functions:
            print("Inform {0}".format(feature_function))
            feature_function.inform(preprocessed)

from torch import nn, autograd,rand
import torch
import sys
if __name__ == "__main__":

    m = nn.Linear(20, 30)
    input = autograd.Variable(torch.rand(12949, 20))
    output = m(input)
    print(output.size())

    sexism_file = os.path.join("data","sexism.json")
    racism_file = os.path.join("data","racism.json")
    neither_file = os.path.join("data","neither.json")
    waseem_hovy = os.path.join("data","amateur_expert.json")


    csvreader = CSVReader(encoding="ISO-8859-1")
    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema())
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema())

    datasets = [
        DataSet(file=sexism_file, reader=jlr, formatter=formatter),
        DataSet(file=racism_file, reader=jlr, formatter=formatter),
        DataSet(file=neither_file, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy, reader=jlr, formatter=formatter2),
        ]

    composite = CompositeDataset()
    for dataset in datasets:
        dataset.read()
        composite.add(dataset)

    features = Features([UnigramFeatureFunction(),
                         BigramFeatureFunction(),
                         CharNGramFeatureFunction(1),
                         CharNGramFeatureFunction(2),
                         CharNGramFeatureFunction(3)
                         ])

    features.generate_vocab(composite)
    fs = features.load(composite)

    model = MLP(fs[0].shape[1],123,composite.num_classes())

    train(model,fs,200,1e-3,10)