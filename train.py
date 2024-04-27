from BiasStudy import datasets, predictionKit
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, EarlyStopping
from BiasStudy.datasets import FairFaceDataset
from pathlib import Path


fair_face_dataset = FairFaceDataset(
    data_dir = "/inputs/fair-face-volume/fairface",
    train_labels_csv_name = "fairface_label_train.csv",
    validation_labels_csv_name = "fairface_label_val.csv",
    under_sample = True,
    image_shape = (224,224,3),
    feature_column = "file",
    output_col = "binary_race",
    overwrite_sample_number = 3000
)

train_df = fair_face_dataset.get_train_pd()
BATCH_SIZE = 64


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation="relu", input_shape=fair_face_dataset.get_image_shape()))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2))

model.compile(
    optimizer="rmsprop",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


train_datagen = ImageDataGenerator(
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(), 
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
    subset = "training"
)


validation_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size = (fair_face_dataset.get_image_width(),fair_face_dataset.get_image_height()),
    x_col = fair_face_dataset.get_feature_col_name(),
    y_col = fair_face_dataset.get_output_col_name(),
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
    subset = "validation"
)

history = []

Path("/outputs/training-output-dataset/checkpoints").mkdir(parents=True, exist_ok=True)
Path("/outputs/training-output-dataset/logging").mkdir(parents=True, exist_ok=True)
Path("/outputs/training-output-dataset/model").mkdir(parents=True, exist_ok=True)
Path("/outputs/training-output-dataset/weights").mkdir(parents=True, exist_ok=True)

filepath="/outputs/training-output-dataset/checkpoints/cp-{epoch:02d}.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor = 'val_accuracy',
    verbose = 1
)

log_csv = CSVLogger(
    filename = "/outputs/training-output-dataset/logging/logs.csv",
    append = True
)

callbacks_list = [checkpoint, log_csv]


history.append(
    model.fit(
        train_generator,
        epochs = 2,
        validation_data = validation_generator,
        callbacks=callbacks_list
    )
)

train_loss, train_acc = model.evaluate(train_generator)
validation_loss, test_acc = model.evaluate(validation_generator)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))