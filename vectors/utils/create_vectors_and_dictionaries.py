import numpy as np
import sys
# from scipy import spatial
# import operator

''' ARGS '''

gen_lan = str(sys.argv[1])
gen_lan_name = str(sys.argv[1])
c1 = str(sys.argv[2])
c1_name = str(sys.argv[2])
c2 = str(sys.argv[3])
c2_name = str(sys.argv[3])
c3 = str(sys.argv[4])
c3_name = str(sys.argv[4])


''' FUNCTIONS '''

# given the outputfile including the embeddings for MAIN, genlan, c1 and c2, create a different list for each of them (the three resulting lists have the same lenght)
def CreateAListForEachCommuinity(emb_file, reddit_dic, c1_dic, c2_dic, c3_dic):

    print('creating a list for each community and writing vocabularies')

    main_list = []
    gen_lan_list = []
    c1_list = []
    c2_list = []
    c3_list = []

    for line in emb_file:

        line = line.replace(',','.')

        if line.split()[0] == 'MAIN':
            main_list.append(np.asarray(line.split()[2:], dtype=float))

        if line.split()[0] == str(gen_lan):
            gen_lan_list.append(np.asarray(line.split()[2:], dtype=float))
            reddit_dic.write(line.split()[1] + '\n')

        if line.split()[0] == str(c1):
            c1_list.append(np.asarray(line.split()[2:], dtype=float))
            c1_dic.write(line.split()[1] + '\n')

        if line.split()[0] == str(c2):
            c2_list.append(np.asarray(line.split()[2:], dtype=float))
            c2_dic.write(line.split()[1] + '\n')

        if line.split()[0] == str(c3):
            c3_list.append(np.asarray(line.split()[2:], dtype=float))
            c3_dic.write(line.split()[1] + '\n')

    # check that the lists are the same lenght

    print(len(main_list))
    print(len(gen_lan_list))
    print(len(c1_list))
    print(len(c2_list))
    print(len(c3_list))
    if len(main_list) == len(gen_lan_list) and len(gen_lan_list) == len(c1_list) and len(c1_list) == len(c2_list) and len(c2_list) == len(c3_list):
        print('all lists are the same lenght')
    else:
        print('there is a problem: lists are not the same lenght')

    print('done with the lists')

    return main_list, gen_lan_list, c1_list, c2_list, c3_list


# create the vector for word i for a specific community by summing the ith vector in main and the ith (deviation) vector in c
def SumMainandDeviationVectors(main, gen_lan, c1, c2, c3, reddit_vectors, c1_vectors, c2_vectors, c3_vectors):

    print('summing main vectors and deviations vectors and writing files')

    gen_lan_main_plus_dev_vec = []
    c1_main_plus_dev_vec = []
    c2_main_plus_dev_vec = []
    c3_main_plus_dev_vec = []

    i = 0

    while i < len(main):

        # write files with vectors
        c1_vectors.write(str(np.add(main[i], c1[i])).replace('\n', '').replace('    ', ' ').replace('   ', ' ').replace('  ',' ').replace('[ ', '').replace(' ]', ' ').strip('[]') + '\n')

        c2_vectors.write(str(np.add(main[i], c2[i])).replace('\n', '').replace('    ', ' ').replace('   ', ' ').replace('  ',' ').replace('[ ', '').replace(' ]', ' ').strip('[]') + '\n')

        c3_vectors.write(str(np.add(main[i], c3[i])).replace('\n', '').replace('    ', ' ').replace('   ', ' ').replace('  ',' ').replace('[ ', '').replace(' ]', ' ').strip('[]') + '\n')

        reddit_vectors.write(str(np.add(main[i], gen_lan[i])).replace('\n', '').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace('[ ', '').replace(' ]', ' ').strip('[]') + '\n')

        i = i+1

    print('done')

    return gen_lan_main_plus_dev_vec, c1_main_plus_dev_vec, c2_main_plus_dev_vec, c3_main_plus_dev_vec


''' IMPORT DATA '''

# embeddings_file = open('/Users/marcodeltredici/workspace/PYCHARM/Community_Lexicon_experiment/local_code_only/results/with_gen_lan_reduced/geoDIST.' + gen_lan + '.' + c1 + '.' + c2 + '.' + 'out.embeddings').readlines()
embeddings_file = open('/Users/marcodeltredici/workspace/PYCHARM/Community_Lexicon_experiment/local_code_only/results/with_gen_lan_reduced/geoDIST.'  + gen_lan + '.' + c1 + '.' + c2 + '.' + c3 + '.out.embeddings').readlines()

reddit_vectors = open('./vectors/vectors.' + gen_lan_name + '.txt', 'w')
c1_vectors = open('./vectors/vectors.' + c1_name + '.txt', 'w')
c2_vectors = open('./vectors/vectors.' + c2_name + '.txt', 'w')
c3_vectors = open('./vectors/vectors.' + c3_name + '.txt', 'w')

reddit_dictionary = open('./dictionaries/dictionary.' + gen_lan_name + '.txt', 'w')
c1_dictionary = open('./dictionaries/dictionary.' + c1_name + '.txt', 'w')
c2_dictionary = open('./dictionaries/dictionary.' + c2_name + '.txt', 'w')
c3_dictionary = open('./dictionaries/dictionary.' + c3_name + '.txt', 'w')


main_list, gen_lan_list, c1_list, c2_list, c3_list = CreateAListForEachCommuinity(embeddings_file, reddit_dictionary, c1_dictionary, c2_dictionary, c3_dictionary)

SumMainandDeviationVectors(main_list, gen_lan_list, c1_list, c2_list, c3_list, reddit_vectors, c1_vectors, c2_vectors, c3_vectors)