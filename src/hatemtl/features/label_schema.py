from common.dataset.label_schema import LabelSchema


class WaseemLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemLabelSchema, self).__init__(["Racism","Sexism","None"])


class WaseemHovyLabelSchema(LabelSchema):
    def __init__(self):
        super(WaseemHovyLabelSchema, self).__init__(["Racism","Sexism","Neither","Both"])


class DavidsonLabelSchema(LabelSchema):
    def __init__(self):
        super(DavidsonLabelSchema, self).__init__(["Hate Speech","Offensive Language","Neither"])


class DavidsonToZLabelSchema(LabelSchema):
    def __init__(self):
        super(DavidsonToZLabelSchema, self).__init__(["Racism","Sexism","Neither"])

