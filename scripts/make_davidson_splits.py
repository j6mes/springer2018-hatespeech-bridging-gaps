"""
Stratified sampling of train/dev/test instances of Davidson dataset
"""

from collections import defaultdict
import random

infile = open("data/davidson.clean.csv")
outfile_tr = open("data/davidson.tr.csv", "w")
outfile_dv = open("data/davidson.dv.csv", "w")
outfile_te = open("data/davidson.te.csv", "w")

class2tweets = defaultdict(list)

for line in infile:
    fields = line.strip().split(",", 7)
    cls = fields[5]
    class2tweets[cls].append(line.strip())

instances_tr = []
instances_dv = []
instances_te = []

for cls, items in class2tweets.items():
    random.shuffle(items)
    n_tr = int(len(items) * 0.8)
    n_dv = int(len(items) * 0.1)
    instances_tr += items[:n_tr]
    instances_dv += items[n_tr:(n_tr+n_dv)]
    instances_te += items[(n_tr + n_dv):]

random.shuffle(instances_tr)
random.shuffle(instances_dv)
random.shuffle(instances_te)

for inst in instances_tr:
    outfile_tr.write(inst+'\n')

for inst in instances_dv:
    outfile_dv.write(inst+'\n')

for inst in instances_te:
    outfile_te.write(inst+'\n')
