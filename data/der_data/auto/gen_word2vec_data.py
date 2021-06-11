import pandas as pd
import gzip
import numpy as np
import nltk



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def gen_word2vec_data(path):
    stop_words = 'stopwords'
    sw = pd.read_csv(stop_words, header=None)
    english_punctuations = ['-', '``', "''", '"', '--', '...', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!',
                            '*', '@', '#', '$', '%']

    stop = []
    #stop = english_punctuations + sw.values[:,0].tolist()
    #stop = english_punctuations

    word_data = []
    for d in parse(path):
        if not pd.isnull(d['reviewText']):
            review = '$$$$&&&&'.join([i.lower() for i in nltk.word_tokenize(d['reviewText']) if i.lower() not in stop])
            #review += '$$$$&&&&||'
            word_data.append(review)
        else:
            print 'meet review=nan'
        
        if not pd.isnull(d['summary']):
            review = '$$$$&&&&'.join([i.lower() for i in nltk.word_tokenize(d['summary']) if i.lower() not in stop])
            #review += '$$$$&&&&||'
            word_data.append(review)
        else:
            print 'meet sum review=nan'


    t = pd.DataFrame(word_data)
    t.to_csv('word_data', header=False, index=None, sep='\t')

    f_in = open('word_data', 'rb')
    f_out = gzip.open('word_data.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


gen_word2vec_data('reviews_Musical_Instruments_5.json.gz')
