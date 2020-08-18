import torch
from torch.autograd import Variable

import numpy as np
import cPickle as pickle
# import pickle as pickle
from collections import deque, Counter
import pandas as pd
from time import strptime
import argparse


data_path='../data/'
save_path='../results/'
data_name='foursquare'
sample_data_name='foursquare_sample'
dataset_path='/home/local/ASUAD/ychen404/Code/DeepMove_new/dataset_tsmc2014/'
dataset_name='dataset_TSMC2014_NYC_simple.txt'
tweets_path = '/home/local/ASUAD/ychen404/Code/DeepMove_new/data/tweets_clean_sample.txt'

# sample_data = pickle.load(open(data_path + sample_data_name + '.pk', 'rb'))
# data = pickle.load(open(data_path + data_name + '.pk', 'rb'))
# print(type(data))
# for k in data:
#     print (k)

# print(sample_data['uid_list'].keys()[0])
# print(sample_data['vid_list'].keys()[0])

# print(data['uid_list']['84202896'])

"""
1. User ID (anonymized)
2. Venue ID (Foursquare)
3. Venue category ID (Foursquare)
4. Venue category name (Fousquare)
5. Latitude
6. Longitude
7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
8. UTC time
"""

# dataset = pd.read_csv(dataset_path + dataset_name, sep="\t", header = None)
# print(dataset.head())

# tweets = pd.read_csv(tweets_path, sep=" ", header=None)
# print(tweets.head())

# Target time format
# 2010-02-25 23:40:19

def print_key_header(data, key):
    i = 0
    for k in data[key].keys():
        print(k)        
        # for v in data['vid_list'][k]:
        #     print(v)
        i += 1
        if i == 5:
            break   

def load_data(dp, dn):
    data = pickle.load(open(dp + dn + '.pk', 'rb'))
    # print(type(data))
    # print("dir of data: {}".format(dir(data)))
    for key in data:
        # print(key)
        cnt = 0
        for value in data[key]:
            cnt += 1
        print("{0} value count: {1}".format(key, cnt))
        
    # print(data['vid_list']['4c807c68f9b79c7404a70c45'])
    
    # print(data['data_filter']['344'])
    cnt_344 = 0
    # for value in data['data_filter']['344']:
    #     cnt_344 += 1
    #     print(len(data['data_filter']['344']))
    # print(data['data_filter']['344']['1'])

    print("data['data_filter']['344'].keys()")
    print(30*"*")
    for k in data['data_filter']['344'].keys():
        print(k)
        
    print(30*"*")
    
    print("sessions_count: {}".format(data['data_filter']['344']['sessions_count']))
    # print("raw_sessions: {}".format(data['data_filter']['344']['raw_sessions']))
    # print("topk: {}".format(data['data_filter']['344']['topk']))
    print("topk_count: {}".format(data['data_filter']['344']['topk_count']))
    # print("sessions: {}".format(data['data_filter']['344']['sessions']))


    # print(data['parameters'])
    i = 0
    counter = 0
    
    # for k in data['vid_list'].keys():
    #     print(k)        
    #     # for v in data['vid_list'][k]:
    #     #     print(v)
    #     i += 1
    #     if i == 5:
    #         break   
    # print_key_header(data, 'vid_list')
    # print_key_header(data, 'vid_lookup')
    # print_key_header(data, 'uid_list')
    # print_key_header(data, 'data_neural')

    # print(data['uid_list'].keys())
    # print(data['data_neural'][0]['train_loc'])

    # print("number of items: {}".format(counter))

def strip():
    with open(dataset_path+dataset_name) as fid:
                for i, line in enumerate(fid):
                    # print(line)
                    
                    uid, vid, _, _, lat, lon, _, tim = line.strip('\r\n').split('\t')
                    # print(new_line[0])
                    # print(uid)
                    # print(vid)
                    # print(tim)
                    day, mon, date, time, zero, year = tim.strip('\r\n').split(' ')
                    mon_num = str(strptime(mon, '%b').tm_mon)
                    # print(year + '-' + mon_num + '-' + date + ' ' + time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--data_name', type=str, default='foursquare')
    args = parser.parse_args()
    print(args)
    argv = vars(args)
    print("data_path: {}".format(argv['data_path']))
    args_dp = argv['data_path']
    args_dn = argv['data_name']
    # print(args.data_name)
    # load_data(args.data_path, args.data_name)
    load_data(args_dp, args_dn)
    # print_key_header('vid_list')

