import re
from pprint import pprint
import json

from dataset_reader import DataSet


def pp_lowercase(text):
    return text.lower()


def pp_strip_hashtags(text):
    return ' '.join(re.sub("(\#[A-Za-z0-9]+)"," # ",text).split())


def pp_strip_usernames(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)"," @ ",text).split())


def pp_strip_url(text):
    return ' '.join(re.sub("(https?\:\/\/[^\s]+)"," U ",text).split())




def preprocess(tweet,tools=list([pp_lowercase,pp_strip_hashtags,pp_strip_usernames, pp_strip_url])):
    text = tweet['text']

    for tool in tools:
        text = tool(text)

    return text



if __name__=="__main__":
    racism = DataSet("Racism")

    for tweet in racism.data:
        print(preprocess(tweet))


