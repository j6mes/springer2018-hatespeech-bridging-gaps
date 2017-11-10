import re
from pprint import pprint
import json

from dataset_reader import DataSet


def pp_lowercase(text):
    return text.lower()


def pp_strip_hashtags(text):
    return ' '.join(re.sub("(\#[A-Za-z0-9]+)"," HASHTAG ",text).split())


def pp_strip_usernames(text):
    return ' '.join(re.sub("(@[A-Za-z0-9\_]+)"," USERNAME ",text).split())


def pp_strip_url(text):
    return ' '.join(re.sub("(https?\:\/\/[^\s]+)"," URL ",text).split())

def pp_replace_numbers(text):
    return re.sub(r'[0-9]+', 'NUMBER', text)

def pp_strip_nonalphanum(text):
    return re.sub(r'[\W\s]+', ' ', text)


def pp_placeholders_singlechar(text):
    text = re.sub('HASHTAG', '#', text)
    text = re.sub('USERNAME', '@', text)
    text = re.sub('URL', '$', text)
    text = re.sub('NUMBER', 'D', text)
    return text




def preprocess(tweet,tools=list([pp_lowercase,
                                 pp_strip_hashtags,
                                 pp_strip_usernames,
                                 pp_strip_url,
                                 pp_strip_nonalphanum,
                                 pp_replace_numbers,
                                 pp_placeholders_singlechar])):
    text = tweet['text']

    for tool in tools:
        text = tool(text)

    return text



if __name__=="__main__":
    racism = DataSet("Racism")

    for tweet in racism.data:
        print(preprocess(tweet))


