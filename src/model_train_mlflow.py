import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Flatten
from keras.models import Model
from glob import glob
import os
import argparse
from get_data import get_data
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import tensorflow
import mlflow
# from disutils.command.config import config
from http import client
# from mlflow.tracking import mlflowClient
from pprint import pprint
from urllib.parse import urlparse


def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        trn_set = config['model']['train_path']
        te_set = config["model"]["test_path"]
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        verticalf = config['img_augment']['vertical_flip']
        horizontalf = config['img_augment']['horizontal_flip']
        batch = config['img_augment']['batch_size']
        class_mode = config['img_augment']['class_mode']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs =  config['model']['epochs']
        print(type(batch))
        resnet = VGG19(input_shape=img_size + [3],weights='imagenet',include_top = False)
        for p in resnet.layers:
            p.trainable = False
        op = Flatten()(resnet.output)
        prediction = Dense(num_cls,activation='softmax')(op)
        mod = Model(inputs = resnet.input,outputs=prediction)
        print(mod.summary())
        img_size = tuple(img_size)
        mod.compile(loss = loss,optimizer = optimizer,metrics = metrics)
        train_gen = ImageDataGenerator(rescale=1./255,shear_range=shear_range,zoom_range=zoom_range,
                                        vertical_flip=verticalf,horizontal_flip = horizontalf,rotation_range = 90)

        test_gen = ImageDataGenerator(rescale=1./255)

        train_set = train_gen.flow_from_directory(trn_set,target_size = (255,255),batch_size = batch,class_mode = class_mode)
        test_set = test_gen.flow_from_directory(te_set,target_size = (255,255),batch_size = batch,class_mode = class_mode)

        ########################### MLFLOW START ###############################################

        mlflow_config = config["mlflow_config"]
        remote_server_uri = mlflow_config["remote_server_uri"]

        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(mlflow_config["experiment_name"])
        
        with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
            history = mod.fit(train_set,epochs=epochs,validation_data=test_set, steps_per_epoch = len(train_set),validation_steps = len(test_set))

            train_loss = history.history["loss"][-1]
            train_acc = history.history["accuracy"][-1]
            val_loss = history.history['val_loss'][-1]

            print("train_loss:", train_loss)
            print("train_acc:", train_acc)
            print("val_loss:", val_loss)

            mlflow.log_param("epochs",epochs)
            mlflow.log_param("train_loss",train_loss)
            mlflow.log_param("train_accuracy",train_acc)
            mlflow.log_param("val_loss",val_loss)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(mod,"model",registered_model_name = mlflow_config["registered_model_name"])
            
            else:
                mlflow.keras.load_model(mod,"model")
        
        ##################################### MLFLOW END #####################################


    else:
        print("model is not trained")



if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    passed_args = args.parse_args()
    train_model_mlflow(config_file=passed_args.config)