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
    racism = DataSet("racism_overlapfree")
    splits = int(len(racism.data) * 0.8), int(len(racism.data) * 0.9)
    train, dev, test = racism.data[:splits[0]], racism.data[splits[0]:splits[1]], racism.data[splits[1]:]
    print(len(racism.data), len(train), len(dev), len(test))

    # for tweet in racism.data:
    #     print(tweet['text'])

    neither = DataSet("neither_overlapfree")
    splits = int(len(neither.data) * 0.8), int(len(neither.data) * 0.9)
    train, dev, test = neither.data[:splits[0]], neither.data[splits[0]:splits[1]], neither.data[splits[1]:]
    print(len(neither.data), len(train), len(dev), len(test))


    sexism = DataSet("sexism_overlapfree")
    splits = int(len(sexism.data) * 0.8), int(len(sexism.data) * 0.9), int(len(sexism.data) * 0.1)
    train, dev, test = sexism.data[:splits[0]], sexism.data[splits[0]:splits[1]], sexism.data[splits[1]:]
    print(len(sexism.data), len(train), len(dev), len(test))

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
