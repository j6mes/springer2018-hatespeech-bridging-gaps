class Formatter():
    def __init__(self,label_schema, preprocessing=None, mapping=None):
        self.label_schema = label_schema
        self.mapping = mapping
        self.preprocessing = preprocessing


    def format(self,lines):
        formatted = []
        for line in lines:
            fl = self.format_line(line)
            if fl is not None:
                if isinstance(fl,list):
                    formatted.extend(fl)
                else:
                    formatted.append(fl)

        if self.mapping is not None:
            for item in formatted:
                item["label"] = self.mapping[item["label"]]

        if self.preprocessing is not None:
            for item in formatted:
                item["data"] = self.preprocessing(item["data"])
        return formatted

    def format_line(self,line):
        pass
