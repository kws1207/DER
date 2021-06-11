import pandas as pd
import numpy as np
import gzip
import pickle
import nltk
import re
import time as tm
from collections import Counter
import itertools


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def process(path):
    whole_data = {}
    sorted_whole_data = {}
    ui_review_sentences = {}
    user_filter_number = 0
    item_filter_number = 0
    min_time = np.inf
    max_time = 0
    time_bin_number = 1000
    stop = []
    stop_words = 'stopwords'
    english_punctuations = ['-', '``', "''", '"', '--', '...', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    sw = pd.read_csv(stop_words, header=None)
    stop += english_punctuations+ list(sw.values[:,0])
    user_purchased_items = dict()
    item_purchased_users = dict()

    tmp = []
    print 'first read begin ...'
    s = tm.time()
    line_num = 0
    for d in parse(path):
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        time = d['unixReviewTime']
        user = d['reviewerID']
        item = d['asin']
        review = d['reviewText']
        if user not in user_purchased_items.keys():
            user_purchased_items[user] = [item]
        else:
            user_purchased_items[user].append(item)

        if item not in item_purchased_users.keys():
            item_purchased_users[item] = [user]
        else:
            item_purchased_users[item].append(user)
        if time > max_time:
            max_time = time
        if time < min_time:
            min_time = time
        review_tokens = [i.lower() for i in nltk.word_tokenize(review) if i.lower() not in stop]
        tmp += review_tokens
    print str(tm.time()-s)

    vocabulary = dict(Counter(tmp))
    interval = (max_time-min_time)/time_bin_number
    max = np.array(vocabulary.values()).max()
    vocabulary = {k:v for k,v in vocabulary.items() if v >= 5 and v <= max*0.95}
    vocabulary = sorted(vocabulary.iteritems(), key=lambda d: d[1], reverse=True)[:50000]
    vocabulary = {i[0]:i[1] for i in vocabulary}
    vocabulary['||'] = 1
    pickle.dump(vocabulary, open('vocabulary', 'wb'))
    print 'first read end ...'

    print 'second read begin ...'
    line_num = 0
    s = tm.time()
    for d in parse(path):
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        user = d['reviewerID']
        item = d['asin']
        rating = d['overall']
        time = d['unixReviewTime']
        review = d['reviewText']
        time = int(((time - min_time) / interval))

        if len(user_purchased_items[user]) > int(user_filter_number) and \
           len(item_purchased_users[item]) > int(item_filter_number):
            review_sentences = [i.strip() for i in re.split(",|\.|:|;|\?", review)]
            review_sentences = [[i.lower() for i in nltk.word_tokenize(rev)
                                 if i.lower() not in stop and i.lower() in list(vocabulary.keys())] for rev in
                                review_sentences]
            review_sentences = [t for t in review_sentences if t != []]
            review_tokens_1 = list(itertools.chain.from_iterable(review_sentences))
            if len(review_sentences) > 0 and len(review_tokens_1) > 0:
                ui_review_sentences[user + '@' + item] = review_sentences
                record = '||'.join([str(item), str(rating), '@'.join(review_tokens_1), str(time)])
                if user not in whole_data.keys():
                    whole_data[user] = [record]
                else:
                    whole_data[user].append(record)
    print str(tm.time() - s)
    pickle.dump(ui_review_sentences, open('ui_review_sentences', 'wb'))

    print len(whole_data.items())
    for user, interactions in whole_data.items():
        sort_index = np.argsort([int(i.split('||')[-1]) for i in interactions])
        sorted_interactions = list(np.array(interactions)[sort_index])
        sorted_whole_data[user] = sorted_interactions
    print len(sorted_whole_data.items())

    pickle.dump(sorted_whole_data, open('named_whole_data', 'wb'))

    print(str(tm.time() - s))
    print 'second read end ...'


    # third read to generate user item dicts
    print 'third read begin ...'
    item_id_dict = dict()
    user_id_dict = dict()
    id_item_dict = dict()
    id_user_dict = dict()
    item_number = 0
    user_number = 0
    sorted_whole_data = pickle.load(open('named_whole_data', 'rb'))

    # generate dicts
    line_num = 0
    for user, line in sorted_whole_data.items():
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        for inter in line:
            item = inter.split('||')[0]
            if user not in user_id_dict.keys():
                user_id_dict[user] = str(user_number)
                id_user_dict[str(user_number)] = user
                user_number += 1
            if item not in item_id_dict.keys():
                item_id_dict[item] = str(item_number)
                id_item_dict[str(item_number)] = item
                item_number += 1

    pickle.dump(user_id_dict, open('user_id_dict', 'wb'))
    pickle.dump(id_user_dict, open('id_user_dict', 'wb'))
    pickle.dump(item_id_dict, open('item_id_dict', 'wb'))
    pickle.dump(id_item_dict, open('id_item_dict', 'wb'))
    print 'third read end ...'


    # 4th read to map real name to id
    print 'fourth read begin ...'
    user_purchased_items = dict()
    item_purchased_users = dict()
    data = dict()
    data_statistics = dict()

    data_statistics['max_interaction_length'] = 0
    data_statistics['interaction_num'] = 0
    data_statistics['user_num'] = 0
    data_statistics['item_num'] = 0
    data_statistics['word_num'] = 0
    data_statistics['time_bin_number'] = time_bin_number
    line_num = 0
    word_id_dict_tmp = pickle.load(open('word_id_dict_tmp', 'r'))
    word_emb_dict_tmp = pickle.load(open('word_emb_tmp.pkl', 'rb'))
    print word_id_dict_tmp[',']
    word_id_dict = dict()
    id_word_dict = dict()
    final_embeddings = []

    word_number = 0
    ui = []
    for user, line in sorted_whole_data.items():
        line_num += 1
        print 'read ' + str(line_num) + ' lines'
        for inter in line:
            inter = inter.split('||')
            item = inter[0]
            ui.append(str(user)+str(item))
            rating = inter[1]
            review = inter[2]
            time = inter[3]

            if user not in user_purchased_items.keys():
                user_purchased_items[user] = [item]
            else:
                user_purchased_items[user].append(item)

            if item not in item_purchased_users.keys():
                item_purchased_users[item] = [user]
            else:
                item_purchased_users[item].append(user)

            if pd.isnull(review) == False:
                review_tokens = review.split('@')
                #review_tokens.append('||')
                review_ids = ''

                #print review_tokens
                for w in review_tokens:
                    if w in word_id_dict_tmp.keys() and w in list(vocabulary.keys()):
                        if w not in word_id_dict.keys():
                            word_id_dict[w] = str(word_number)
                            id_word_dict[str(word_number)] = w
                            final_embeddings.append(word_emb_dict_tmp[word_id_dict_tmp[w]])
                            word_number += 1
                        review_ids += word_id_dict[w] + '::'
                    else:
                        print 'not in big dict or vocabulary'

                review_ids = review_ids[:-2]
                record = '||'.join([str(item_id_dict[item]), str(rating), str(review_ids), str(time)])

                if user_id_dict[user] not in data.keys():
                    data[user_id_dict[user]] = [record]
                else:
                    data[user_id_dict[user]].append(record)
            else:
                print 'fffffff'

    for user, items in user_purchased_items.items():
        if data_statistics['max_interaction_length'] < len(items):
            data_statistics['max_interaction_length'] = len(items)
            print user_purchased_items[user]

    data_statistics['user_num'] = len(user_id_dict.items())
    data_statistics['item_num'] = len(item_id_dict.items())
    data_statistics['word_num'] = len(word_id_dict.items())
    data_statistics['interaction_num'] = len(list(set(ui)))
    print 'user number :' + str(data_statistics['user_num'])
    print 'item number :' + str(data_statistics['item_num'])
    print 'word number :' + str(data_statistics['word_num'])
    print 'interaction number :' + str(data_statistics['interaction_num'])
    print 'max interaction length :' + str(data_statistics['max_interaction_length'])


    pickle.dump(word_id_dict, open('word_id_dict', 'wb'))
    pickle.dump(id_word_dict, open('id_word_dict', 'wb'))
    pickle.dump(final_embeddings, open('word_emb.pkl', "wb"))

    pickle.dump(user_purchased_items, open('user_purchased_items', 'wb'))
    pickle.dump(item_purchased_users, open('item_purchased_users', 'wb'))
    pickle.dump(data, open('ided_whole_data', 'wb'))
    pickle.dump(data_statistics, open('data_statistics', 'wb'))
    print 'fourth read end ...'

process('reviews_Musical_Instruments_5.json.gz')

