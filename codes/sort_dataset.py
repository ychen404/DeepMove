import argparse
import random
from sparse_traces import DataFoursquare, parse_args

# FILE = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_user_1.txt'
# OUTPUT = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_user_1_top100.txt'

FILE = '../dataset_tsmc2014/dataset_TSMC2014_NYC.txt'
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
    count_dict = {}
    for l in readable:
        uid = l.split('\t')[0]
        if uid not in count_dict:
            count_dict[uid] = 1
        else:
            count_dict[uid] += 1

    # print(count_dict)
    cnt_keys = 0
    for k, v in count_dict.items():
        cnt_keys += 1
    print("The total number of keys is: {}".format(cnt_keys))
    return count_dict

with open(FILE) as f:
    # sort the lines based on uid (column 1), and column 1 is type int
    
    unique_dict = {}
    sorted_lines = sortLinesByColumn(f, 1, int)
    unique_dict = count_unique_uid(sorted_lines)

    # dict is ordered so change to list and change back
    key_array = list(unique_dict.items())
    random.shuffle(key_array)

    # store the first uids in an array
    elememnt_count = 0
    # key_array_top_100 = []

    print("The length of the array: {}".format(len(key_array)))

def sample_users(array, pub_uid, pri_uid):
    # select 100 users from a randomly shuffled input
    for x in array:
        if x not in pub_uid and len(pub_uid) < 100:
            pub_uid.append(x)
            # print(x)
        else:
            pri_uid.append(x)
    return pub_uid, pri_uid

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
        count_unique_uid(f)

args = parse_args()
data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                hour_gap=args.hour_gap, min_gap=args.min_gap,
                                session_min=args.session_min, session_max=args.session_max,
                                sessions_min=args.sessions_min, train_split=args.train_split, save_name=args.save_name)
parameters = data_generator.get_parameters()
print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
print('############START PROCESSING:')
print('load trajectory from {}'.format(data_generator.DATASET_PATH + data_generator.DATASET_NAME))
data_generator.load_trajectory_from_tweets()
print('filter users')
filtered_uid = data_generator.filter_users_by_length()
print(len(filtered_uid))
# unique(filtered_uid)

random.shuffle(filtered_uid)
public_uid, private_uid = sample_users(filtered_uid, public_uid, private_uid)
print("The length of the public_uid: {}".format(len(public_uid)))
print("The length of the private_uid: {}".format(len(private_uid)))
# print(top_100(filtered_uid))

with open(FILE, 'r') as fd:
    with open(OUTPUT_PUB, 'w') as fout_pub:
        with open(OUTPUT_PRI, 'w') as fout_pri:
            for line in fd:
                l = line.split('\t')
                if l[0] in public_uid:
                    fout_pub.write(line)
                elif l[0] in private_uid:
                    fout_pri.write(line)

verify(FILTERED)
verify(FILE)
verify(OUTPUT_PUB)
verify(OUTPUT_PRI)