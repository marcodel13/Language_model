vectors = open('./vectors/vectors.soccer.txt').readlines()

for line in vectors:
    if len(line.split()) != 200:
        print('error!')