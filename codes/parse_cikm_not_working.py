import argparse
import random
from time import strptime
from collections import Counter
import cPickle as pickle
import time

"""
This code creates the number of datasets based on the number of effective users from the tweets-cikm dataset
Somehow the filter user result alway returns zero
Started from scratch from the original sparse_traces.py
"""

class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72, min_gap=10, session_min=2, session_max=10,
                 sessions_min=2, train_split=0.8, embedding_len=50, save_name='cikm_private', 
                 dataset_name='tweets-cikm'):
        tmp_path = "../data/"
        
        self.SAVE_PATH = tmp_path
        self.save_name = save_name
        self.DATASET_PATH='/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/'
        # self.DATASET_NAME='dataset_TSMC2014_NYC_20000.txt'
        # self.DATASET_NAME='dataset_TSMC2014_NYC_20000_user_1.txt'
        # self.DATASET_NAME='dataset_TSMC2014_NYC_private.txt'
        self.DATASET_NAME = dataset_name
        # self.DATASET_NAME='dataset_TSMC2014_NYC_20000_top100.txt'


        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.train_split = train_split

        self.data = {}
        self.venues = {}
        self.words_original = []
        self.words_lens = []
        self.dictionary = dict()
        self.words_dict = None
        self.data_filter = {}
        self.user_filter3 = None
        self.uid_list = {}
        self.vid_list = {'unk': [0, -1]}
        self.vid_list_lookup = {}
        self.vid_lookup = {}
        self.pid_loc_lat = {}
        self.data_neural = {}

    # ############# 1. read trajectory data from twitters
    def load_trajectory_from_tweets(self):
        with open(self.DATASET_PATH + self.DATASET_NAME + '.txt') as fid:
        # with open(self.TWITTER_PATH, 'r') as fid:

            _, cnt_keys = count_unique_uid(fid)
            print("There are {} unique users in the dataset".format(cnt_keys))
            for i, line in enumerate(fid):
                _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                print("uid:{}".format(uid))
                
                ########## Match the fields in Foursquare NYC dataset ##########
                # print(i)
                # if i % 1000 == 0:
                #     print(i, line)

                # uid, pid, _, _, _, _, _, tim_orig = line.strip('\r\n').split('\t')

                ########## Convert character month to number to match with the clean_tweets_sample.txt ########## 
                # day, mon, date, time, zero, year = tim_orig.strip('\r\n').split(' ')
                # mon_num = str(strptime(mon, '%b').tm_mon)
                # tim = (year + '-' + mon_num + '-' + date + ' ' + time)
                
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])
                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1

    # # ########### 3.0 basically filter users based on visit length and other statistics
    # def filter_users_by_length(self):
    #     """
    #     [ expression for item in list if conditional ]
    #     """
# ########### 3.0 basically filter users based on visit length and other statistics
    def filter_users_by_length(self):
        uid_3 = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        pick3 = sorted([(x, len(self.data[x])) for x in uid_3], key=lambda x: x[1], reverse=True)
        pid_3 = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        pid_pic3 = sorted([(x, self.venues[x]) for x in pid_3], key=lambda x: x[1], reverse=True)
        pid_3 = dict(pid_pic3)

        session_len_list = []
        for u in pick3:
            uid = u[0]
            info = self.data[uid]
            topk = Counter([x[0] for x in info]).most_common()
            topk1 = [x[0] for x in topk if x[1] > 1]
            sessions = {}
            for i, record in enumerate(info):
                poi, tmd = record
                try:
                    tid = int(time.mktime(time.strptime(tmd, "%Y-%m-%d %H:%M:%S")))
                except Exception as e:
                    print('error:{}'.format(e))
                    continue
                sid = len(sessions)
                if poi not in pid_3 and poi not in topk1:
                    # if poi not in topk1:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                        sessions[sid] = [record]
                    elif (tid - last_tid) / 60 > self.min_gap:
                        sessions[sid - 1].append(record)
                    else:
                        pass
                last_tid = tid
            sessions_filter = {}
            for s in sessions:
                if len(sessions[s]) >= self.filter_short_session:
                    sessions_filter[len(sessions_filter)] = sessions[s]
                    session_len_list.append(len(sessions[s]))
            if len(sessions_filter) >= self.sessions_count_min:
                self.data_filter[uid] = {'sessions_count': len(sessions_filter), 'topk_count': len(topk), 'topk': topk,
                                         'sessions': sessions_filter, 'raw_sessions': sessions}

        self.user_filter3 = [x for x in self.data_filter if
                             self.data_filter[x]['sessions_count'] >= self.sessions_count_min]
                             
        filtered_user_id = self.user_filter3
        return filtered_user_id

    # ########### 4. build dictionary for users and location
    def build_users_locations_dict(self):
        for u in self.user_filter3:
            sessions = self.data_filter[u]['sessions']
            if u not in self.uid_list:
                self.uid_list[u] = [len(self.uid_list), len(sessions)]
            for sid in sessions:
                poi = [p[0] for p in sessions[sid]]
                for p in poi:
                    if p not in self.vid_list:
                        self.vid_list_lookup[len(self.vid_list)] = p
                        self.vid_list[p] = [len(self.vid_list), 1]
                    else:
                        self.vid_list[p][1] += 1

    # support for radius of gyration
    def load_venues(self):
        # with open(self.TWITTER_PATH, 'r') as fid:
        with open(self.DATASET_PATH + self.DATASET_NAME, 'r') as fid:
            for line in fid:
                _, uid, lon, lat, tim, _, _, tweet, pid = line.strip('\r\n').split('')
                # uid, pid, _, _, lat, lon, _, tim_orig = line.strip('\r\n').split('\t')
                
                # day, mon, date, time, zero, year = tim_orig.strip('\r\n').split(' ')
                # mon_num = str(strptime(mon, '%b').tm_mon)
                # tim = (year + '-' + mon_num + '-' + date + ' ' + time)
                self.pid_loc_lat[pid] = [float(lon), float(lat)]

    def venues_lookup(self):
        for vid in self.vid_list_lookup:
            pid = self.vid_list_lookup[vid]
            lon_lat = self.pid_loc_lat[pid]
            self.vid_lookup[vid] = lon_lat

    # ########## 5.0 prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        tm = time.strptime(tmd, "%Y-%m-%d %H:%M:%S")
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    def prepare_neural_data(self):
        for u in self.uid_list:
            sessions = self.data_filter[u]['sessions']
            sessions_tran = {}
            sessions_id = []
            for sid in sessions:
                sessions_tran[sid] = [[self.vid_list[p[0]][0], self.tid_list_48(p[1])] for p in
                                      sessions[sid]]
                sessions_id.append(sid)
            split_id = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            pred_len = sum([len(sessions_tran[i]) - 1 for i in train_id])
            # print("pred_len: {}".format(pred_len))
            valid_len = sum([len(sessions_tran[i]) - 1 for i in test_id])
            # print("valid_len: {}".format(valid_len))
            train_loc = {}
            for i in train_id:
                for sess in sessions_tran[i]:
                    if sess[0] in train_loc:
                        train_loc[sess[0]] += 1
                    else:
                        train_loc[sess[0]] = 1
            # calculate entropy
            entropy = entropy_spatial(sessions)

            # calculate location ratio
            train_location = []
            for i in train_id:
                train_location.extend([s[0] for s in sessions[i]])
            train_location_set = set(train_location)
            test_location = []
            for i in test_id:
                test_location.extend([s[0] for s in sessions[i]])
            test_location_set = set(test_location)
            whole_location = train_location_set | test_location_set
            test_unique = whole_location - train_location_set
            location_ratio = len(test_unique) / len(whole_location)

            # calculate radius of gyration
            lon_lat = []
            for pid in train_location:
                try:
                    lon_lat.append(self.pid_loc_lat[pid])
                except:
                    print(pid)
                    print('error')
            lon_lat = np.array(lon_lat)
            center = np.mean(lon_lat, axis=0, keepdims=True)
            center = np.repeat(center, axis=0, repeats=len(lon_lat))
            rg = np.sqrt(np.mean(np.sum((lon_lat - center) ** 2, axis=1, keepdims=True), axis=0))[0]

            self.data_neural[self.uid_list[u][0]] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id,
                                                     'pred_len': pred_len, 'valid_len': valid_len,
                                                     'train_loc': train_loc, 'explore': location_ratio,
                                                     'entropy': entropy, 'rg': rg}

    # ############# 6. save variables
    def get_parameters(self):
        parameters = {}
        # parameters['TWITTER_PATH'] = self.TWITTER_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['train_split'] = self.train_split

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'vid_list': self.vid_list, 'uid_list': self.uid_list,
                              'parameters': self.get_parameters(), 'data_filter': self.data_filter,
                              'vid_lookup': self.vid_lookup}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    parser.add_argument('--global_visit', type=int, default=10, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    parser.add_argument('--save_name', type=str, default='foursquare_nyc_20000_user_1', help="the file name for the pk")
    parser.add_argument('--dataset_name', type=str, default='dataset_TSMC2014_NYC_private', help="the file name for the txt")

    return parser.parse_args()
# FILE = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_user_1.txt'
# OUTPUT = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_user_1_top100.txt'


# FILE = '../dataset_tsmc2014/dataset_TSMC2014_NYC.txt'
FILE = '../serm-data/tweets-cikm.txt'
# OUTPUT = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_top100.txt'
FILTERED = '../dataset_tsmc2014/dataset_TSMC2014_NYC_filtered.txt'
OUTPUT_PUB = '../dataset_tsmc2014/dataset_TSMC2014_NYC_public.txt'
OUTPUT_PRI = '../dataset_tsmc2014/dataset_TSMC2014_NYC_private.txt'

NUM_WORKERS = 2
random.seed(10)
worker_bin = [[] for i in range(NUM_WORKERS)]

# Use the first 100 users as public 
public_uid = []
private_uid = []
filtered_uid = []

def sortLinesByColumn(readable, column, column_type):
    """Returns a list of strings (lines in readable file) in sorted order (based on column)"""
    lines = []

    for i, line in enumerate(readable):
        # get the element in column based on which the lines are to be sorted
        if i % 1000 == 0:
            print(i, line)
        
        column_element= column_type(line.split('\t')[column-1])
        lines.append((column_element, line))

    lines.sort()

    return [x[1] for x in lines]

# parser = argparse.ArgumentParser()
# parser.add_argument("echo", help="echo the string")
# parser.add_argument("num_workers", help="number of workers")

# args = parser.parse_args()
# print(args.echo)

# Initalize arrays to hold user data

def count_unique_uid(readable):
    # print_l = True
    count_dict = {}
    for l in readable:
        # if print_l == True:
        #     print(l)
        #     print_l = False
        _, uid, _, _, tim, _, _, tweet, pid = l.strip('\r\n').split('')
        # uid = l.strip('\r\n').split('')[1]

        # uid = l.strip('\r\n').split(' ')[1]
        # print(uid)
        if uid not in count_dict:
            count_dict[uid] = 1
        else:
            count_dict[uid] += 1

    # print(count_dict)
    cnt_keys = 0
    for k, v in count_dict.items():
        cnt_keys += 1
    print("The total number of keys is: {}".format(cnt_keys))
    return count_dict, cnt_keys

# with open(FILE) as f:
#     # sort the lines based on uid (column 1), and column 1 is type int
    
#     unique_dict = {}
#     sorted_lines = sortLinesByColumn(f, 1, int)
#     unique_dict, _ = count_unique_uid(sorted_lines)

#     # dict is ordered so change to list and change back
#     key_array = list(unique_dict.items())
#     random.shuffle(key_array)

#     # store the first uids in an array
#     elememnt_count = 0
#     # key_array_top_100 = []

#     print("The length of the array: {}".format(len(key_array)))

def sample_users(array, pub_uid, pri_uid):
    # select 100 users from a randomly shuffled input
    for x in array:
        if x not in pub_uid and len(pub_uid) < 100:
            pub_uid.append(x)
            # print(x)
        else:
            pri_uid.append(x)
    return pub_uid, pri_uid



def sample_users_dynamic(array, sample_size):
    # Split the private uids into two part to use as local data    
    # select users from a randomly shuffled input
    uid_grp_1, uid_grp_2 = [], []
    
    for x in array:
        if x not in uid_grp_1 and len(uid_grp_1) < sample_size:
            uid_grp_1.append(x)
            # print(x)
        else:
            uid_grp_2.append(x)
    return uid_grp_1, uid_grp_2

def unique(list1): 

    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    print("The length of the unique list is: {}".format(len(unique_list)))
    # for x in unique_list: 
    #     print x

def verify(path):
    with open(path, 'r') as f:
        _, cnt = count_unique_uid(f)
    return cnt

args = parse_args()
data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                hour_gap=args.hour_gap, min_gap=args.min_gap,
                                session_min=args.session_min, session_max=args.session_max,
                                sessions_min=args.sessions_min, train_split=args.train_split, save_name=args.save_name, dataset_name=args.dataset_name)
parameters = data_generator.get_parameters()
print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
print('############START PROCESSING:')
print('load trajectory from {}'.format(data_generator.DATASET_PATH + data_generator.DATASET_NAME))
data_generator.load_trajectory_from_tweets()
print('filter users')
filtered_uid = data_generator.filter_users_by_length()
print("There are {} unique filtered users".format(len(filtered_uid)))
# unique(filtered_uid)

random.shuffle(filtered_uid)
public_uid, private_uid = sample_users(filtered_uid, public_uid, private_uid)

# print("The length of the public_uid: {}".format(len(public_uid)))
# print("The length of the private_uid: {}".format(len(private_uid)))
# print(top_100(filtered_uid))

# with open(FILE, 'r') as fd:
#     with open(OUTPUT_PUB, 'w') as fout_pub:
#         with open(OUTPUT_PRI, 'w') as fout_pri:
#             for line in fd:
#                 l = line.split('\t')
#                 if l[0] in public_uid:
#                     fout_pub.write(line)
#                 elif l[0] in private_uid:
#                     fout_pri.write(line)


# verify(FILTERED)
# verify(FILE)
# verify(OUTPUT_PUB)
print("The number of keys in cikm_20000: {}".format(verify(FILE)))

# uid_grp_1, uid_grp_2 = sample_users_dynamic(private_uid, 714/2)