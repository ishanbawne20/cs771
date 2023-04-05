infile = open("dict",'r')
outfile = open("dict_secret",'w')

lines = infile.readlines()

lines = [x[:-1] for x in lines]

for line in lines:
    for word in line.split():
        outfile.write(word)
        outfile.write("\n")