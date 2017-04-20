from pprint import pprint
import json

class DataSet():

    def __init__(self,name,base_dir ="data/",ext="json",lines=True):
        with open(base_dir+name+"."+ext,"r") as f:
            if lines:
                self.data = []
                for line in f.readlines():
                    self.data.append(json.loads(line.strip()))
            else:
                self.data = json.load(f)


if __name__=="__main__":
    racism = DataSet("Racism")
    print(len(racism.data))


    neither = DataSet("neither")
    print(len(neither.data))


    sexism = DataSet("sexism")
    print(len(sexism.data))

