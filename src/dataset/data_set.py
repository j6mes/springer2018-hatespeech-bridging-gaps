import os

from dataset.formatter import TextAnnotationFormatter
from dataset.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema
from dataset.reader import JSONLineReader


class DataSet():
    def __init__(self,file,reader,formatter):
        self.reader = reader
        self.file = file
        self.formatter = formatter
        self.data = []


    def read(self):
        self.data.extend(self.formatter.format(self.reader.read(self.file)))



class CompositeDataset():
    def __init__(self):
        self.data = []

    def add(self,dataset):
        print("Adding "+ str(dataset))
        self.data.extend(dataset.data)

if __name__ == "__main__":
    sexism_file = os.path.join("data","sexism.json")
    racism_file = os.path.join("data","racism.json")
    neither_file = os.path.join("data","neither.json")
    waseem_hovy = os.path.join("data","amateur_expert.json")


    jlr = JSONLineReader()
    formatter = TextAnnotationFormatter(WaseemLabelSchema())
    formatter2 = TextAnnotationFormatter(WaseemHovyLabelSchema())


    datasets = [
        DataSet(file=sexism_file, reader=jlr, formatter=formatter),
        DataSet(file=racism_file, reader=jlr, formatter=formatter),
        DataSet(file=neither_file, reader=jlr, formatter=formatter),
        DataSet(file=waseem_hovy, reader=jlr, formatter=formatter2)
        ]

    composite = CompositeDataset()
    for dataset in datasets:
        dataset.read()
        print(dataset.data[0:10])
        composite.add(dataset)


    print(composite.data)
