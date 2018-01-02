from common.dataset.formatter import Formatter


class TextAnnotationFormatter(Formatter):
    def format_line(self,line):
        return {"data":line["text"],"label":self.label_schema.get_id(line["Annotation"])}

class DavidsonFormatter(Formatter):
    def format_line(self, line):
        return {"data": line["tweet_text"], "label": self.label_schema.get_id(line["does_this_tweet_contain_hate_speech"])}