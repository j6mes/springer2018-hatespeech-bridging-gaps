
class LabelSchema:
    def __init__(self,labels):
        self.labels = {self.preprocess(val):idx for idx,val in enumerate(labels)}
        self.idx = {idx:self.preprocess(val) for idx,val in enumerate(labels)}

    def get_id(self,label):
        if self.preprocess(label) in self.labels:
            return self.labels[self.preprocess(label)]
        return None

    def preprocess(self,item):
        return item.lower()



class WaseemLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemLabelSchema, self).__init__(["None","Racism","Sexism"])


class WaseemHovyLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemHovyLabelSchema, self).__init__(["Neither","Racism","Sexism","Both"])


class DavidsonLabelSchema(LabelSchema):
    def __init__(self):
        super(DavidsonLabelSchema, self).__init__(["Neither","Offensive","HateSpeech"])

