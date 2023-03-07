import argparse
import os
import shutil # For File System File
import logging # For Exception Handling
import yaml
import boto3
import pandas as pd
import numpy as np
from get_data import get_data

import boto3
client = boto3.client('s3')
root_dir = 's3://mlops-data-course5i13/'

def train_test_split(config):
    config = get_data(config)
    # root_dir = config['base']['data_source']
    dest = config['load_data']['preprocessed_data']
    p  = config['load_data']['full_path']
    cla = config['base']['data_source']
    # cla = os.listdir(cla)
    # print(root_dir)

    splittr = config['train_split']['split_ratio']
    for k in range(len(cla)):
        per = len(os.listdir(os.path.join(root_dir,cla[k])))
        print(k,"->",per)
        cnt = 0
        split_ratio = round((splittr/100)*per)
        for j in os.listdir((os.path.join(root_dir + '/' + cla[k],j))):
            if (cnt!=split_ratio):
                shutil.copy(path,dest + '/'+'train/class_' + str(k))
                cnt+=1
            else:
                shutil.copy(path,dest + '/'+'train/class_' + str(k))
    
        print('Done')















if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    passed_args = args.parse_args()
    train_test_split(config=passed_args.config)