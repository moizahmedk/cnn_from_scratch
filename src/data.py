import tensorflow as tf
import numpy as np
from datasets import load_dataset

IMG_SIZE = 64
BATCH_SIZE = 32


def load_eurosat():
    """
    Load EuroSAT RGB dataset from Hugging Face.
    """
    ds = load_dataset("blanchon/EuroSAT_RGB")
    class_names = ds["train"].features["label"].names
    num_classes = len(class_names)
    return ds, class_names, num_classes


def preprocess(example):
    """
    Convert PIL image → Tensor, normalize, resize.
    MUST return a dict for Hugging Face compatibility.
    """
    image = np.array(example["image"])
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return {
        "image": image,
        "label": example["label"]
    }


def build_tf_datasets(ds):
    """
    Convert Hugging Face dataset → tf.data.Dataset
    """
    train_ds = ds["train"].with_transform(preprocess).to_tf_dataset(
        columns="image",
        label_cols="label",
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    val_ds = ds["validation"].with_transform(preprocess).to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=BATCH_SIZE
    )

    test_ds = ds["test"].with_transform(preprocess).to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds, test_ds
