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


def train(epochs=30, validation_split=0.2, save_path="models/cnn.keras"):
    X, y = load_augmented_data()
    X_test, y_test = load_raw_test_data()

    model = build_cnn(input_shape=(32, 32, 3), num_classes=len(CARD_LABELS))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("=" * 50)
    print("ARHITEKTURA MODELA:")
    print("=" * 50)
    model.summary()

    print("\n" + "=" * 50)
    print("TRENIRANJE:")
    print("=" * 50)

    # Keras fit() funkcija koristi shuffle kao default ali zbog reproducibilnosti randomizujemo podatke sami sa zadatim seedom
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    history = model.fit(
        X, y,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1,
    )

    print("\n" + "=" * 50)
    print("EVALUACIJA:")
    print("=" * 50)

    # Test na augmentovanim podacima
    aug_loss, aug_acc = model.evaluate(X, y, verbose=0)
    print(f"Augmentovani dataset - loss: {aug_loss:.4f}, accuracy: {aug_acc:.4f}")

    # Test na neaugmentovanim originalima
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neaugmentovani test  - loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    # Cuvanje modela
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel sacuvan u {save_path}")

    return model, history


if __name__ == "__main__":
    train()