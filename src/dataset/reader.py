import csv
import json
import random

from util.random import get_seed


class Reader:
    def __init__(self,encoding="utf-8"):
        self.enc = encoding

    def read(self,file):
        with open(file,"r",encoding = self.enc) as f:
            return self.process(f)

    def process(self,f):
        pass


class CSVReader(Reader):
    def process(self,fp):
        r = csv.DictReader(fp)
        return [line for line in r]

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




