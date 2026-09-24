import os

import numpy as np
import tensorflow as tf

from src.build_dataset import CARD_LABELS, RAW_DIR
from src.cnn import build_cnn
from src.helper import encode_images, load_images_from_folder


def load_augmented_data():
    """Ucitava augmentovani dataset (X, y) iz data/encoded/."""
    X = np.load(os.path.join("data", "encoded", "X.npy"))
    y = np.load(os.path.join("data", "encoded", "y.npy"))

    # X je (N, 3072) - vracamo u oblik slike (N, 32, 32, 3) za CNN
    X = X.reshape(-1, 32, 32, 3)
    return X, y


def load_raw_test_data():
    """Ucitava neaugmentovani test dataset (X, y) iz data/raw/."""
    images, labels = [], []
    for card, label in CARD_LABELS.items():
        folder = os.path.join(RAW_DIR, card)
        raw = load_images_from_folder(folder)
        encoded = encode_images(raw)
        images.append(encoded)
        labels.extend([label])
    X = np.concatenate(images).reshape(-1, 32, 32, 3)
    y = np.array(labels)
    return X, y
