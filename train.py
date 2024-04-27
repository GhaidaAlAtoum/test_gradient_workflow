import sys
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_config_path', type=str, required=True)
parser.add_argument('-f', '--fair_face_path', default='/inputs/fair-face-volume/fairface', type=str)
parser.add_argument('-e', '--num_epochs', default=32, type=int, required=True)
parser.add_argument('-b', '--batch_size', default=32, type=int, required=True)
parser.add_argument('-s', '--early_stopping_patience', default=10, type=int, required=True)
parser.add_argument('-c', '--checkpoint_frequencey', default=1, type=int, required=True)
parser.add_argument('--overwrite_sample_number', default=False, type=bool)
parser.add_argument('--number_samples', required='--overwrite_sample_number' in sys.argv ,default=100, type=int)


global args
args = parser.parse_args()

from BiasStudy import datasets, predictionKit
from BiasStudy.datasets import FairFaceDataset
import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, EarlyStopping
from keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


def load_config_and_validate(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    if "model" not in config:
        raise Exception("Missing field 'model'")
        
    if "model_name" not in config["model"]:
        raise Exception("Missing field 'model.model_name'")
        
    if "conv_blocks" not in config["model"]:
        raise Exception("Missing field 'model.conv_blocks'")
    
    return config

def create_model(
    num_classes: int,
    config_dict: dict,
    image_shape: Tuple[int, int, int] = (224,224,3)
):
    model_name = config_dict['model']['model_name']
    cnn_model = keras.Sequential(name=model_name)
    cnn_model.add(Input(shape=image_shape))
    input = Input(shape = image_shape)
    for block_num, (block_key, block_config) in enumerate(config_dict['model']['conv_blocks'].items()):
        for conv_layer_num in range(0, block_config['num_conv_layers']):
            cnn_model.add(
                Conv2D(
                    filters = block_config['num_filters'],
                    kernel_size = block_config['kernel_size'],
                    padding = 'same',
                    activation = 'relu',
                    name = "block{}_conv{}".format(block_num, conv_layer_num)
                )
            )
        cnn_model.add(MaxPool2D(pool_size =2, strides =2, padding ='same', name="block{}_pool".format(block_num)))
    cnn_model.add(Flatten(name = "flatten"))
    
    if "flatt_layers" in config_dict['model']:
        for flat_layer_num, (flat_key, flat_config) in enumerate(config_dict['model']['flatt_layers'].items()):
            cnn_model.add(Dense(units = flat_config['num_units'], activation ='relu'))

    cnn_model.add(Dense(units = num_classes, activation ='softmax', name = "prediction"))
    return cnn_model

def compile_model(num_classes: int, cnn_model: keras.Model):
    loss = None
    if num_classes > 1:
        loss = "categorical_crossentropy"
    else:
        loss = "mean_squared_error"
    cnn_model.compile(
        optimizer = Adam(learning_rate=0.001),
        loss = loss,
        metrics = ['accuracy']
    )

# https://stackoverflow.com/a/72746245
class EpochModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self,
                 filepath,
                 frequency=1,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 options=None,
                 **kwargs):
        super(EpochModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                   mode, "epoch", options)
        self.epochs_since_last_save = 0
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.epochs_since_last_save % self.frequency == 0:
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        pass

def create_dirs(model_name: str):
    checkpoint_dir = "./outputs/{}/checkpoints/".format(model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logging_dir = "./outputs/{}/logging/".format(model_name)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    
    model_dir = "./outputs/{}/model/".format(model_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    weight_dir = "./outputs/{}/weights/".format(model_name)
    Path(weight_dir).mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir, logging_dir, model_dir, weight_dir

def train_model(
    early_stopping_patience: int,
    cnn_model: keras.Model,
    train_generator: DataFrameIterator,
    validation_generator: DataFrameIterator,
    num_epochs: int,
    checkpoint_frequencey: int = 1,
):
    checkpoint_dir, logging_dir, model_dir, weight_dir = create_dirs(cnn_model.name)
    
    checkpoint_file_path = checkpoint_dir + "cp-{epoch:02d}.ckpt"
    checkpoint_callback = EpochModelCheckpoint(
        filepath = checkpoint_file_path,
        monitor = 'val_accuracy',
        frequencey = checkpoint_frequencey,
        verbose = 1
    )
    
    log_csv_file_path = logging_dir + "logs.csv"
    log_csv_callback = CSVLogger(
        filename = log_csv_file_path,
        append = True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=early_stopping_patience, 
        mode='min',
        min_delta=0.0001
    )

    callbacks_list = [checkpoint_callback, log_csv_callback, early_stopping]
    
    cnn_model.fit(
        train_generator,
        epochs = num_epochs,
        validation_data = validation_generator,
        callbacks = callbacks_list
    )
    
    train_loss, train_acc = cnn_model.evaluate(train_generator)
    validation_loss, test_acc = cnn_model.evaluate(validation_generator)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    cnn_model.save(model_dir + "model.h5".format(cnn_model.name))
    cnn_model.save_weights(weight_dir)

print("-------------- Loading Model Config from {}".format(args.model_config_path))

config_dict = load_config_and_validate(args.model_config_path)

print("-------------- Reading Dataset from {}".format(args.fair_face_path))

fair_face_dataset = FairFaceDataset(
    data_dir = args.fair_face_path,
    train_labels_csv_name = "fairface_label_train.csv",
    validation_labels_csv_name = "fairface_label_val.csv",
    under_sample = True,
    image_shape = (224,224,3),
    feature_column = "file",
    output_col = "binary_race",
    overwrite_sample_number = args.number_samples
)

train_df = fair_face_dataset.get_train_pd()

train_datagen = ImageDataGenerator(
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(), 
    batch_size = args.batch_size,
    class_mode = "categorical",
    subset = "training"
)


validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(),
    batch_size = args.batch_size,
    class_mode = "categorical",
    subset = "validation"
)

print("-------------- Create Model and Compile")
cnn = create_model(num_classes = 2, config_dict = config_dict)
compile_model(num_classes = 2, cnn_model = cnn)

cnn.summary()

print("-------------- Train Model")

train_model(
    cnn_model = cnn,
    train_generator = train_generator,
    validation_generator = validation_generator,
    num_epochs = args.num_epochs,
    early_stopping_patience = args.early_stopping_patience,
    checkpoint_frequencey = args.checkpoint_frequencey
)