import numpy as np
import pandas as pd
import random
import codecs
import pickle
from collections import OrderedDict


trainfile = 'review_dict'
wordidmapfile = "wordidmapfile"
thetafile =  "thetafile"
phifile = "phifile"
paramfile = "paramfile"
topNfile = "topNfile"
tassginfile = "tassginfile"

K = 5
alpha = 50.0 / K + 1
beta = 0.01
iter_times = 500
top_words_num = 20


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")


class LDAModel(object):
    def __init__(self, dpre):

        self.dpre = dpre

        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num

        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile

        self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")
        self.nwsum = np.zeros(self.K, dtype="int")
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")
        self.Z = np.array(
            [[0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])

        for x in xrange(len(self.Z)):
            self.ndsum[x] = self.dpre.docs[x].length
            for y in xrange(self.dpre.docs[x].length):
                topic = random.randint(0, self.K - 1)
                self.Z[x][y] = topic
                self.nw[self.dpre.docs[x].words[y]][topic] += 1
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1

        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])
        self.phi = np.array([[0.0 for y in xrange(self.dpre.words_count)] for x in xrange(self.K)])

    def sampling(self, i, j):

        topic = self.Z[i][j]
        word = self.dpre.docs[i].words[j]
        self.nw[word][topic] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1

        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        self.p = (self.nw[word] + self.beta) / (self.nwsum + Vbeta) * \
                 (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)
        for k in xrange(1, self.K):
            self.p[k] += self.p[k - 1]

        u = random.uniform(0, self.p[self.K - 1])
        for topic in xrange(self.K):
            if self.p[topic] > u:
                break

        self.nw[word][topic] += 1
        self.nwsum[topic] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1

        return topic

    def est(self):
        for x in xrange(self.iter_times):
            print ("iteration: %s" % x)
            for i in xrange(self.dpre.docs_count):
                for j in xrange(self.dpre.docs[i].length):
                    topic = self.sampling(i, j)
                    self.Z[i][j] = topic
        print("iteration complete")
        print("compute doc-topic distribution")
        self._theta()
        print("compute word-topic distrobution")
        self._phi()
        print("save model")
        self.save()

    def _theta(self):
        for i in xrange(self.dpre.docs_count):
            self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)

    def _phi(self):
        for i in xrange(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)

    def save(self):
        print("doc-topic distribution has been saved to %s" % self.thetafile)
        with codecs.open(self.thetafile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        print("word-topic distribution has been saved to %s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')

        print("parameters have been saved to %s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write('iter_times=' + str(self.iter_times) + '\n')
            f.write('top_words_num=' + str(self.top_words_num) + '\n')

        print("topN words have been saved to %s" % self.topNfile)

        with codecs.open(self.topNfile, 'w') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in xrange(self.K):
                f.write('number' + str(x) + '\n')
                twords = []
                twords = [(n, self.phi[x][n]) for n in xrange(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')

        print("doc-word-topic allocations have been saved to %s" % self.tassginfile)
        with codecs.open(self.tassginfile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
                f.write('\n')
        print("model training complete")


def preprocessing(dict_path):
    print('loading data......')
    dict = pickle.load(open(dict_path, 'r'))

    print("loading complete...")
    dpre = DataPreProcessing()
    items_idx = 0
    doc_index = []

    for k, line in dict.items():
        if dict_path == 'user_dict':
            for time, review in line.items():
                l = ' '.join(review)
                if l != "":
                    doc_index.append(k + '@'+ str(time))
                    tmp = l.strip().split()
                    doc = Document()
                    for item in tmp:
                        if dpre.word2id.has_key(item):
                            doc.words.append(dpre.word2id[item])
                        else:
                            dpre.word2id[item] = items_idx
                            doc.words.append(items_idx)
                            items_idx += 1
                    doc.length = len(tmp)
                    dpre.docs.append(doc)
                else:
                    print k, line
                    pass

        else:
            line = ' '.join(line)
            if line != "":
                doc_index.append(k)
                tmp = line.strip().split()
                doc = Document()
                for item in tmp:
                    if dpre.word2id.has_key(item):
                        doc.words.append(dpre.word2id[item])
                    else:
                        dpre.word2id[item] = items_idx
                        doc.words.append(items_idx)
                        items_idx += 1
                doc.length = len(tmp)
                dpre.docs.append(doc)
            else:
                print k, line
                pass

    dpre.docs_count = len(dpre.docs)
    dpre.words_count = len(dpre.word2id)
    print("%s doc" % dpre.docs_count)
    dpre.cachewordidmap()
    print("word-id has been saved to %s" % wordidmapfile)
    return dpre, doc_index


def run():
    path = 'user_dict'
    dpre, doc_index = preprocessing(path)
    lda = LDAModel(dpre)
    lda.est()
    topic_dict = {}
    theta = pd.read_csv('thetafile', header=None)
    num = 0
    for line in theta.values:
        index = doc_index[num]
        user = index.split('@')[0]
        time = index.split('@')[1]
        if user not in topic_dict.keys():
            topic_dict[user] = [[time, line[0].split('\t')[:-1]]]
        else:
            topic_dict[user].append([time, line[0].split('\t')[:-1]])
        num += 1
    pickle.dump(topic_dict, open('user_topic_dict', 'wb'))


    path = 'time_dict'
    dpre, doc_index = preprocessing(path)
    lda = LDAModel(dpre)
    lda.est()
    print len(doc_index)
    topic_dict = {}
    theta = pd.read_csv('thetafile', header=None)
    num = 0
    for line in theta.values:
        index = doc_index[num]
        topic_dict[index] = line[0].split('\t')[:-1]
        num += 1
    pickle.dump(topic_dict, open('time_topic_dict', 'wb'))



if __name__ == '__main__':
    doc_index = run()
    #t = pickle.load(open('time_topic_dict', 'r'))
    #for k,v in t.items():
    #    print k, v
    #    raw_input()
