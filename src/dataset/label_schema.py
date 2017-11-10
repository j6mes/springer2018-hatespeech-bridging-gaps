
class LabelSchema:
    def __init__(self,labels):
        self.labels = {val:idx for idx,val in enumerate(labels)}
        self.idx = {idx:val for idx,val in enumerate(labels)}


class WaseemLabelSchema(LabelSchema):
    def __init__(self,extra=list()):
        labels = ["NotOffensive","Racism","Sexism"]
        labels.extend(extra)
        super(WaseemLabelSchema, self).__init__(labels)


class WaseemHovyLabelSchema(WaseemLabelSchema):
    def __init__(self):
        super(WaseemHovyLabelSchema, self).__init__(["Both"])


class DavidsonLabelSchema(LabelSchema):
    def __init__(self):
        super(DavidsonLabelSchema, self).__init__(["NotOffensive","Offensive","HateSpeech"])

