import json
import random

import os

from dataset.label_schema import WaseemLabelSchema
from util.random import get_seed


class Reader:
    def read(self,file):
        with open(file,"r") as f:
            return self.process(f)

    def process(self,f):
        pass


class CSVReader(Reader):
    pass


class JSONReader(Reader):
    def process(self,fp):
        return json.load(fp)


class JSONLineReader(Reader):
    def process(self,fp):
        data = []
        for line in fp.readlines():
            data.append(json.loads(line.strip()))
        return data









class DataSplit():
    r = random.Random(get_seed())

    def __init__(self,data,train=0.8,dev=0.1):
        self.data = data
        self.r.shuffle(self.data)

        splits = int(len(self.data) * train), int(len(self.data) * (train+dev))
        self.train, self.dev, self.test = self.data[:splits[0]], self.data[splits[0]:splits[1]], self.data[splits[1]:]




