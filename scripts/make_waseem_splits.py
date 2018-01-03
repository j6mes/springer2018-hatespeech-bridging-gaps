"""
Stratified sampling of train/dev/test instances of Davidson dataset
"""

from collections import defaultdict
import random

infile_r = open("data/racism_overlapfree.json")
infile_s = open("data/sexism_overlapfree.json")
infile_n = open("data/neither_overlapfree.json")

labels = ['r', 's', 'n']

outfiles_tr = {lbl: open("data/waseem_{}.tr.json".format(lbl), "w")
               for lbl in labels}
outfiles_dv = {lbl: open("data/waseem_{}.dv.json".format(lbl), "w")
               for lbl in labels}
outfiles_te = {lbl: open("data/waseem_{}.te.json".format(lbl), "w")
               for lbl in labels}

class2tweets = defaultdict(list)


for line in infile_r:
    class2tweets['r'].append(line.strip())

for line in infile_s:
    class2tweets['s'].append(line.strip())

for line in infile_n:
    class2tweets['n'].append(line.strip())


instances_tr = {lbl: [] for lbl in labels}
instances_dv = {lbl: [] for lbl in labels}
instances_te = {lbl: [] for lbl in labels}

for lbl in labels:
    print("Label", lbl)
    items = class2tweets[lbl]
    random.shuffle(items)
    n_tr = int(len(items) * 0.8)
    n_dv = int(len(items) * 0.1)
    instances_tr[lbl] += items[:n_tr]
    instances_dv[lbl] += items[n_tr:(n_tr+n_dv)]
    instances_te[lbl] += items[(n_tr + n_dv):]

    random.shuffle(instances_tr[lbl])
    random.shuffle(instances_dv[lbl])
    random.shuffle(instances_te[lbl])

    for inst in instances_tr[lbl]:
        outfiles_tr[lbl].write(inst+'\n')

    for inst in instances_dv[lbl]:
        outfiles_dv[lbl].write(inst+'\n')

    for inst in instances_te[lbl]:
        outfiles_te[lbl].write(inst+'\n')
