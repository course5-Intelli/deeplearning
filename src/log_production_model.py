import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Input
from keras.models import Model
from glob import glob
import os
import argparse
from get_data import get_data
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from distutils.command.config import config
from http import client
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib

def log_production_model(config_path):
    config = get_data(config_path)
    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    runs = mlflow.search_runs([0])
    lowest = runs["metrics.train_accuracy"].sort_values(ascending = True)
    lowest_run_id = runs[runs["metrics.train_accuracy"]==lowest]["run_id"][0]
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"]==lowest_run_id:
            current_version = mv["Version"]
            logged_model = mv["source"]
            pprint(mv,indent = 4)
            client.transition_model_version_stage(name = model_name,version = current_version,stage = "Production")
        else:
            current_version = mv["Version"]
            client.transition_model_version_stage(name = model_name,version = current_version,stage = "Staging")

    
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = config["model_dir"]
    joblib.dump(loaded_model,model_path)






if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    passed_args = args.parse_args()
    log_production_model(config_path=passed_args.config)