import random

from features import features
from preprocessing import preprocess
from tqdm import tqdm
from vocab import Vocab

from old.dataset_reader import DataSet


class CompositeDataset():

    def __init__(self):
        self.labels = None
        self.data = dict()
        self.train = dict()
        self.dev = dict()
        self.test = dict()
        self.vocab = None

    def add_data(self,class_name,dataset,ff=features,pf=preprocess):
        self.data[class_name] = dataset

        self._import(self.train,class_name,dataset.train,ff,pf)
        self._import(self.dev,class_name,dataset.dev,ff,pf)
        self._import(self.test,class_name,dataset.test,ff,pf)


    def _import(self,target,class_name,data,ff,pf):
        target[class_name] = []

        for item in tqdm(data,desc="Generating features for " + class_name):
            target[class_name].append(ff(pf(item)))

    def get_num_classes(self):
        return len(self.data)

    def label_id(self,label):
        return self.labels.index(label)

    def get_as_labelled(self):
        if self.labels is None:
            self.labels = list(self.data.keys())

            self.final_train = []
            self.final_dev = []
            self.final_test = []

            self.vocab = Vocab()

            for label in self.labels:
                label_id = self.label_id(label)
                self.final_train.extend([(a, label_id) for a in self.train[label]])
                self.final_dev.extend([(a, label_id) for a in self.dev[label]])
                self.final_test.extend([(a, label_id) for a in self.test[label]])

                self.vocab.add(self.train[label])

            self.vocab.generate_dict()
            r = random.Random(1234)
            r.shuffle(self.final_train)

        return self.final_train,self.final_dev,self.final_test,self.vocab

if __name__ == "__main__":
    racism = DataSet("racism_overlapfree")
    sexism = DataSet("sexism_overlapfree")
    neither = DataSet("neither_overlapfree")

    data = CompositeDataset()
    data.add_data('racism',racism)
    data.add_data('sexism',sexism)
    data.add_data('neither',neither)

    print(len(data.get_as_labelled()))