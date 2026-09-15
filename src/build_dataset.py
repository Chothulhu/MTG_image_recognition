"""Pipeline: raw slike -> augmentacija -> enkodovanje -> dataset."""

import os

import numpy as np

from src.augment import augment_image_x_times
from src.helper import (
    encode_images,
    load_images_from_folder,
    save_encoded_images,
    save_pil_images,
)

RAW_DIR = os.path.join("data", "raw")
AUGMENTED_DIR = os.path.join("data", "augmented")
ENCODED_DIR = os.path.join("data", "encoded")

SIZE = (32, 32)
AUGMENT_PER_IMAGE = 10

CARD_LABELS = {
    "angelic_renewal": 0,
    "ironshell_beetle": 1,
    "jade_avenger": 2,
    "soulherder": 3,
    "thallid": 4,
}


def build_dataset(
    raw_dir=RAW_DIR,
    size=SIZE,
    augment_per_image=AUGMENT_PER_IMAGE,
):
    """Gradi augmentovani dataset i vraca (X, y).

    Za svaku kartu: ucitava raw slike, augmentuje ih, snima
    augmentovane verzije (puna rezolucija) i enkodovane verzije (32x32).
    Na kraju snima X.npy i y.npy u data/encoded/.
    """
    X, y = [], []

    for card, label in CARD_LABELS.items():
        folder = os.path.join(raw_dir, card)
        raw_images = load_images_from_folder(folder)

        augmented = []
        for img in raw_images:
            augmented.extend(augment_image_x_times(img, augment_per_image))

        save_pil_images(augmented, os.path.join(AUGMENTED_DIR, card), card)

        encoded = encode_images(augmented, size)
        save_encoded_images(encoded, os.path.join(ENCODED_DIR, card), card, size)

        X.append(encoded)
        y.extend([label] * len(encoded))

    X = np.vstack(X)
    y = np.array(y)

    np.save(os.path.join(ENCODED_DIR, "X.npy"), X)
    np.save(os.path.join(ENCODED_DIR, "y.npy"), y)

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")