import os


class DataSet():
    def __init__(self,file,reader,formatter):
        self.reader = reader
        self.file = file
        self.formatter = formatter
        self.data = []

    def read(self):
        if os.getenv("DEBUG","").lower() in ["1","y","yes","t"]:
            self.data.extend(filter(lambda record: record is not None, self.formatter.format(self.reader.read(self.file)[:10])))
        else:
            self.data.extend(filter(lambda record: record is not None, self.formatter.format(self.reader.read(self.file))))

    def num_classes(self):
        return len(self.formatter.label_schema.labels)

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