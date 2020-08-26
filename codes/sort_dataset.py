import argparse
import random

FILE = '../dataset_tsmc2014/dataset_TSMC2014_NYC_20000_user_1.txt'
NUM_WORKERS = 2
random.seed(10)

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
worker_bin = [[] for i in range(NUM_WORKERS)]

# Use the first 100 users as public 
public_uid = []

private_uid = []


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

    # random shuffle the uids
    # dict is ordered so change to list and change back
    key_array = list(unique_dict.items())
    # print(30*'*' + 'Before shuffling' + 30*'*')
    # print("{}".format(key_array))
    
    random.shuffle(key_array)
    # print(30*'*' + 'After shuffling' + 30*'*')
    # print("{}".format(key_array))

    # somehow the dict always maintain a same order in python 2.7
    # shuffle_dict = dict(tmp_list)
    
    # print(30*'*' + 'shuffle dict' + 30*'*')
    # print(shuffle_dict)
    
    # store the first uids in an array
    elememnt_count = 0
    key_array_top_100 = []

    print("The length of the array: {}".format(len(key_array)))
    # for x in key_array:
    #     # print(x[0])

    for x in key_array:
        if x not in key_array_top_100 and len(key_array_top_100) < 100:
            key_array_top_100.append(x[0])
        elif len(key_array_top_100) >= 100:
            break
        else: 
            continue
    
    print(key_array_top_100, len(key_array_top_100))
        
    # Iterate through the array and save the corresponding entry in the dataset