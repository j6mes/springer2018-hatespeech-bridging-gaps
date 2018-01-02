from common.dataset.label_schema import LabelSchema


class WaseemLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemLabelSchema, self).__init__(["None","Racism","Sexism"])


class WaseemHovyLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemHovyLabelSchema, self).__init__(["Neither","Racism","Sexism","Both"])


class DavidsonLabelSchema(LabelSchema):
    def __init__(self):
        super(DavidsonLabelSchema, self).__init__(["The tweet is not offensive","The tweet uses offensive language but not hate speech","The tweet contains hate speech"])

