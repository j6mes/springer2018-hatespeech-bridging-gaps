

def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output


def char_ngrams(input, n):
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output


def feature_word_bigrams(data):
    return map(lambda text: ["_".join(ng) for ng in ngrams(" ".join(text.split()),2)], data)


def feature_char_unigrams(data):
    return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 1)],data)


def feature_char_bigrams(data):
    return map(lambda text:["".join(ng) for ng in char_ngrams(" ".join(text.split()), 2)],data)


def feature_char_trigrams(data):
    return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 3)],data)





def feature_word_unigrams(data):
    return map(lambda text: [w for w in " ".join(text.split()).split()], data)


def feature_word_bigrams(data):
    return map(lambda text: ["_".join(ng) for ng in ngrams(" ".join(text.split()),2)], data)


def feature_char_unigrams(data):
    return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 1)],data)


def feature_char_bigrams(data):
    return map(lambda text:["".join(ng) for ng in char_ngrams(" ".join(text.split()), 2)],data)


def feature_char_trigrams(data):
    return map(lambda text: ["".join(ng) for ng in char_ngrams(" ".join(text.split()), 3)],data)



