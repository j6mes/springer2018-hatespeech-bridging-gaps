class Formatter():
    def __init__(self,label_schema):
        self.label_schema = label_schema

    def format(self,lines):
        formatted = []
        for line in lines:
            formatted.append(self.format_line(line))
        return formatted

    def format_line(self,line):
        pass



class TextAnnotationFormatter(Formatter):
    def format_line(self,line):
        return {"data":line["text"],"annotation":self.label_schema.get_id(line["Annotation"])}

class DavidsonFormatter(Formatter):
    def format_line(self, line):
        return {"data": line["tweet_text"], "annotation": self.label_schema.get_id(line["does_this_tweet_contain_hate_speech"])}