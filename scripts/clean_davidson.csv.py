import re

infile = open("data/davidson.csv")
outfile = open("data/davidson.clean.csv", "w")

item_id = 0

for i, line in enumerate(infile):
    line = line.strip()
    if i == 0:
        outfile.write(line)
    else:
        # if line.startswith(str(item_id)+","):
        if re.search('^\s*[0-9]+,', line):
            outfile.write("\n")
            outfile.write(line)
            item_id += 1
        else:
            outfile.write(" "+line)

