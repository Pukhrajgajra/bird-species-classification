import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

DATA_DIR = os.path.expanduser("~/bird_classification/data/raw/CUB_200_2011")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 200

def load_dataset_info():
    images_file = os.path.join(DATA_DIR, "images.txt")
    labels_file = os.path.join(DATA_DIR, "image_class_labels.txt")
    split_file  = os.path.join(DATA_DIR, "train_test_split.txt")
    classes_file = os.path.join(DATA_DIR, "classes.txt")

    images_df = pd.read_csv(images_file, sep=" ",
                            header=None, names=["img_id", "filepath"])
    labels_df = pd.read_csv(labels_file, sep=" ",
                            header=None, names=["img_id", "label"])
    split_df  = pd.read_csv(split_file, sep=" ",
                            header=None, names=["img_id", "is_train"])
    classes_df = pd.read_csv(classes_file, sep=" ",
                             header=None, names=["class_id", "class_name"])

    df = images_df.merge(labels_df, on="img_id").merge(split_df, on="img_id")
    df["label"] = df["label"] - 1
    df["full_path"] = df["filepath"].apply(
        lambda x: os.path.join(DATA_DIR, "images", x)
    )
    df["label_str"] = df["label"].apply(
        lambda x: classes_df.iloc[x]["class_name"]
    )

    train_df = df[df["is_train"] == 1].reset_index(drop=True)
    test_df  = df[df["is_train"] == 0].reset_index(drop=True)

    print(f"Training samples:   {len(train_df)}")
    print(f"Test samples:       {len(test_df)}")
    print(f"Number of classes:  {NUM_CLASSES}")
    print(f"Sample classes:     {classes_df['class_name'].head(5).tolist()}")

    return train_df, test_df, classes_df

def create_data_generators(train_df, test_df):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="full_path",
        y_col="label_str",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="full_path",
        y_col="label_str",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="full_path",
        y_col="label_str",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    train_df, test_df, classes_df = load_dataset_info()
    train_gen, val_gen, test_gen = create_data_generators(train_df, test_df)
    print(f"\nBatch shape: {train_gen[0][0].shape}")
    print("Data loader ready!")