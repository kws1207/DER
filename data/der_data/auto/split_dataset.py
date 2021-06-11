import pandas as pd
import numpy as np
import pickle
import gzip

ided_data = pickle.load(open('ided_whole_data', 'r'))

train_user_purchased_items = dict()
train_item_purchased_users = dict()

validation_user_purchased_items = dict()
validation_item_purchased_users = dict()

test_user_purchased_items = dict()
test_item_purchased_users = dict()


train_data = []
validation_data = []
test_data = []
train_validation_data = []

train_items = []

for user, line in ided_data.items():
    listline = [i.split('||') for i in line]

    train_line = listline[:-2]
    for record_index in range(1,len(train_line)):
        item = listline[record_index][0]
        train_items.append(item)
        if user not in train_user_purchased_items.keys():
            train_user_purchased_items[user] = [item]
        else:
            train_user_purchased_items[user].append(item)

        if item not in train_item_purchased_users.keys():
            train_item_purchased_users[item] = [user]
        else:
            train_item_purchased_users[item].append(user)

        if len(train_line[:record_index]) > 0:
            next_times = np.array([int(i) for i in np.array(listline)[1:record_index + 1, 3]])
            pre_times = np.array([int(i) for i in np.array(listline)[:record_index, 3]])
            times = np.array([[str(i)] for i in next_times - pre_times])
            f = np.hstack((np.array(listline)[:record_index, :3], times))
            f = ['||'.join(i) for i in f]

            feature = '()'.join(f)
            target = line[record_index]
            r = '&&'.join([user, feature, target])
            #print feature
            #print target
            #raw_input()
            train_data.append(r)
            train_validation_data.append(r)

for user, line in ided_data.items():
    if len(line[:-2]) > 0:
        listline = [i.split('||') for i in line]
        item = listline[-2][0]
        if item in train_items:
            if user not in validation_user_purchased_items.keys():
                validation_user_purchased_items[user] = [item]
            else:
                validation_user_purchased_items[user].append(item)

            if item not in validation_item_purchased_users.keys():
                validation_item_purchased_users[item] = [user]
            else:
                validation_item_purchased_users[item].append(user)
            next_times = np.array([int(i) for i in np.array(listline)[1:-1, 3]])
            pre_times = np.array([int(i) for i in np.array(listline)[:-2, 3]])
            times = np.array([[str(i)] for i in next_times - pre_times])
            f = np.hstack((np.array(listline)[:-2, :3], times))
            f = ['||'.join(i) for i in f]
            feature = '()'.join(f)
            target = line[-2]
            r = '&&'.join([user, feature, target])
            # print 'validation'
            # print feature
            # print target
            # raw_input()
            validation_data.append(r)
            train_validation_data.append(r)
    else:
        print 'ff'

for user, line in ided_data.items():
    if len(line[:-1]) > 0:
        listline = [i.split('||') for i in line]
        item = listline[-1][0]
        if item in train_items:
            if user not in test_user_purchased_items.keys():
                test_user_purchased_items[user] = [item]
            else:
                test_user_purchased_items[user].append(item)

            if item not in test_item_purchased_users.keys():
                test_item_purchased_users[item] = [user]
            else:
                test_item_purchased_users[item].append(user)
            next_times = np.array([int(i) for i in np.array(listline)[1:, 3]])
            pre_times = np.array([int(i) for i in np.array(listline)[:-1, 3]])
            times = np.array([[str(i)] for i in next_times - pre_times])
            f = np.hstack((np.array(listline)[:-1, :3], times))
            f = ['||'.join(i) for i in f]
            feature = '()'.join(f)
            target = line[-1]
            r = '&&'.join([user, feature, target])
            # print 'test'
            # print feature
            # print target
            # raw_input()
            test_data.append(r)
    else:
        print 'ff'


pickle.dump(train_user_purchased_items, open('train_user_purchased_items', 'wb'))
pickle.dump(train_item_purchased_users, open('train_item_purchased_users', 'wb'))
pickle.dump(test_user_purchased_items, open('test_user_purchased_items', 'wb'))
pickle.dump(test_item_purchased_users, open('test_item_purchased_users', 'wb'))
pickle.dump(validation_user_purchased_items, open('validation_user_purchased_items', 'wb'))
pickle.dump(validation_item_purchased_users, open('validation_item_purchased_users', 'wb'))



t = pd.DataFrame(train_data)
t.to_csv('train_ided_whole_data', index=False, header=None)
t = pd.DataFrame(test_data)
t.to_csv('test_ided_whole_data', index=False, header=None)
t = pd.DataFrame(validation_data)
t.to_csv('validation_ided_whole_data', index=False, header=None)
t = pd.DataFrame(train_validation_data)
t.to_csv('train_validation_ided_whole_data', index=False, header=None)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def review_embedding(path):
    ui_review_sentences = pickle.load(open('ui_review_sentences', 'rb'))
    id_user_dict = {v:k for k,v in pickle.load(open('user_id_dict', 'rb')).items()}
    id_item_dict = {v:k for k,v in pickle.load(open('item_id_dict', 'rb')).items()}
    word_id_dict = pickle.load(open('word_id_dict', 'rb'))
    data_statistics = pickle.load(open('data_statistics', 'rb'))
    data_statistics['max_sentence_length'] = 0
    data_statistics['max_sentence_word_length'] = 0
    emb = pickle.load(open('word_emb.pkl', "rb"))
    stop = []
    stop_words = 'stopwords'
    english_punctuations = ['-', '``', "''", '"', '--', '...', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!',
                            '*', '@', '#', '$', '%']
    sw = pd.read_csv(stop_words, header=None)
    stop += english_punctuations + list(sw.values[:, 0])

    item_reviews = dict()
    user_item_review = dict()
    l = []

    ided_data = pickle.load(open('ided_whole_data', 'r'))

    user_item_real_review = dict()
    item_real_reviews = dict()

    for d in parse(path):
        user = d['reviewerID']
        item = d['asin']
        review = d['reviewText']
        key = user + '@' + item
        user_item_real_review[key] = review
    pickle.dump(user_item_real_review, open('user_item_real_review', 'wb'))

    for user_id, line in ided_data.items():
        line = [i.split('||') for i in line]
        train_line = line[:-1]
        for entry in train_line:
            item_id = entry[0]
            key = id_user_dict[user_id] + '@' + id_item_dict[item_id]
            if key in ui_review_sentences.keys():
                review = ui_review_sentences[key]
                review_embedding = [[emb[int(word_id_dict[i.lower()])] for i in rev] for rev in review]
                review_id_embedding = [[int(word_id_dict[i.lower()]) for i in rev] for rev in review]
                l += [len(rev) for rev in review]
                if len(review) > 0:
                    user_item_review[user_id + '@' + item_id] = [np.array(i).mean(axis=0) for i in review_embedding]
                    r = user_item_real_review[id_user_dict[user_id] + '@' + id_item_dict[item_id]]
                    if item_id not in item_reviews.keys():
                        item_reviews[item_id] = [[user_id, review, [list(i) for i in review_id_embedding]]]
                        item_real_reviews[item_id] = [[user_id, r]]
                    else:
                        item_reviews[item_id].append([user_id, review, [list(i) for i in review_id_embedding]])
                        item_real_reviews[item_id].append([user_id, r])


    for k, v in item_reviews.items():
        if len(v) > data_statistics['max_sentence_length']:
            data_statistics['max_sentence_length'] = len(v)

    data_statistics['max_sentence_word_length'] = np.array(l).max()
    print data_statistics['max_sentence_word_length']
    print data_statistics['max_sentence_length']
    pickle.dump(data_statistics, open('data_statistics', 'wb'))
    pickle.dump(user_item_review, open('user_item_review', 'wb'))
    pickle.dump(item_reviews, open('item_reviews', 'wb'))
    pickle.dump(item_real_reviews, open('item_real_reviews', 'wb'))


review_embedding('reviews_Musical_Instruments_5.json.gz')
