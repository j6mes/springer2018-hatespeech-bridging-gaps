import os

from dataset.formatter import TextAnnotationFormatter, DavidsonFormatter
from dataset.label_schema import WaseemLabelSchema, WaseemHovyLabelSchema, DavidsonLabelSchema
from dataset.reader import JSONLineReader, CSVReader


class DataSet():
    def __init__(self,file,reader,formatter):
        self.reader = reader
        self.file = file
        self.formatter = formatter
        self.data = []


    def read(self):
        self.data.extend(self.formatter.format(self.reader.read(self.file)[:10]))



class CompositeDataset(DataSet):
    def __init__(self):
        self.data = []

    def add(self,dataset):
        print("Adding "+ str(dataset))
        self.data.extend(dataset.data)

    def num_classes(self):

        classes = set()

        for datum in self.data:
            classes.add(datum["label"])

        return len(classes)

if __name__ == "__main__":
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


    davidson = DataSet(file=os.path.join("data","twitter-hate-speech-classifier-DFE-a845520.csv"),
                       reader=csvreader,
                       formatter=DavidsonFormatter(DavidsonLabelSchema()))

    davidson.read()


