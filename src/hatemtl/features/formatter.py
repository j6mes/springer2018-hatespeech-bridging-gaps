from common.dataset.formatter import Formatter


class TextAnnotationFormatter(Formatter):
    def format_line(self,line):
        return {"data":line["text"],"label":self.label_schema.get_id(line["Annotation"])}

class DavidsonFormatter(Formatter):
    def format_line(self, line):
        return {"data": line["tweet"], "label": int(line["class"])}