comm = 'LiverpoolFC'

vec = open('/Users/marcodeltredici/workspace/PYCHARM/LangaugeModelPy/vectors/vectors/vectors.'+comm+'.txt').readlines()
dic = open('/Users/marcodeltredici/workspace/PYCHARM/LangaugeModelPy/vectors/dictionaries/dictionary.'+comm+'.txt').readlines()
out = open('../vec_and_dic_'+comm+'.txt','w')

i = 0

while i < len(dic):

    out.write(dic[i].replace('\n','') + ' ' + vec[i])

    i = i+1