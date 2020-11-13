# For python 3.6.9
import time
import argparse
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import os
import pdb
import time

class DataCrawdad(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.building_lookup = {}
        self.data = []
        self.venues = {}

    def load_trajectory_from_data(self):
        with open(self.data_path) as fid:
            for i, line in enumerate(fid):
                epoch_time, location = line.strip('\r\n').split('\t')
                tim = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(epoch_time)))

                if location == 'OFF':
                    continue
                else:
                    # print(f"time: {tim}, loc: {location}")
                    self.data.append([location, tim])
                    self.building_lookup[i] = location
                if location not in self.venues:
                    self.venues[location] = 1
                else:
                    self.venues[location] += 1

    def filter_users_by_length(self):
        pass
    
                    
    def get_parameters(self):
        parameters = {}
        parameters['data_path'] = self.data_path
        return parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='raw_data/movement_2/2001-2004/sample.mv', help="path to the raw dataset")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_generator = DataCrawdad(args.data_path)
    parameters = data_generator.get_parameters()
    print('PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    data_generator.load_trajectory_from_data()
    print(f"data: {data_generator.data}")
    print(f"venues: {data_generator.venues}")