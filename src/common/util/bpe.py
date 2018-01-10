from math import log


class BPETransformer:
    """
    Transform text to predefined vocabulary of BPE subword units
    Based on https://stackoverflow.com/questions/
              8870261/how-to-split-text-without-spaces-into-list-of-words
    """

    def __init__(self, vocab_file, beginning_of_word="▁"):
        """
        Initialize the transformer with a vocabulary
        :param vocab_file: file containing a whitespace-separated list
        of allowed vocab items, should be ranked by frequency in descending
        order
        :param beginning_of_word: symbol used to indicate a new word
        (default: ▁ [note this is not a normal underscore but Unicode
        Character 'LOWER ONE EIGHTH BLOCK' (U+2581)])
        """
        self.beginning_of_word = beginning_of_word
        self.vocab = open(vocab_file).read().split()
        self.wordcost = dict(
            (k, log((i + 1) * log(len(self.vocab))))
            for i, k in enumerate(self.vocab))
        self.maxword = max(len(x) for x in self.vocab)

    def transform_batch(self, data):
        """
        Transforms a list of text items
        :param data:
        :return:
        """
        for item in data:
            item["data"] = self.transform(item["data"])
            yield item

    def transform(self, text):
        """Uses dynamic programming to infer the location of spaces in a string
           without spaces."""

        text = self.beginning_of_word + text
        text = self.beginning_of_word.join(text.split())

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self.maxword):i]))
            return min(
                (c + self.wordcost.get(text[i - k - 1:i], 9e999), k + 1)
                for k, c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1, len(text) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(text)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            out.append(text[i - k:i])
            i -= k

        return " ".join(reversed(out))

    def untransform(self, text):
        return text.replace(" ", "").replace(self.beginning_of_word, " ")[1:]
