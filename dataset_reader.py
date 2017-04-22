from pprint import pprint
import json
import random

class DataSet():
    r = random.Random(12343)

    def __init__(self,name,base_dir ="data/",ext="json",lines=True,train=0.8,dev=0.1):
        with open(base_dir+name+"."+ext,"r") as f:
            if lines:
                self.data = []
                for line in f.readlines():
                    self.data.append(json.loads(line.strip()))
            else:
                self.data = json.load(f)

        self.r.shuffle(self.data)

        splits = int(len(self.data) * train), int(len(self.data) * (train+dev))
        self.train, self.dev, self.test = self.data[:splits[0]], self.data[splits[0]:splits[1]], self.data[splits[1]:]





if __name__=="__main__":
    racism = DataSet("racism_overlapfree")
    neither = DataSet("neither_overlapfree")
    sexism = DataSet("sexism_overlapfree")


    # Read Expert annotations
    expert = DataSet('amateur_expert')
    e_racism, e_sexism, e_neither, e_both = [], [], [], []

    for item in expert.data:
        if item['Annotation'].lower() == 'sexism':
            e_sexism.append(item)
        elif item['Annotation'].lower() == 'racism':
            e_racism.append(item)
        elif item['Annotation'].lower() == 'neither':
            e_neither.append(item)
        else:
            # TODO Consider merging this into 'racism' to boost counts there
            e_both.append(item)
    print(len(e_racism), len(e_neither), len(e_sexism), len(e_both))
