import csv
import os

import numpy as np

from src.build_dataset import CARD_LABELS
from src.helper import encode_images, load_images_from_folder

MODEL_PATH = os.path.join("models", "cnn.keras")
TEST_DIR = os.path.join("data", "test")
RESULTS_PATH = os.path.join("models", "predictions.csv")

# obrnut rečnik: labela (0-3) -> ime karte
LABEL_TO_CARD = {label: card for card, label in CARD_LABELS.items()}


def load_model(path=MODEL_PATH):
    from tensorflow.keras.models import load_model

    return load_model(path)


def predict_directory(model, test_dir=TEST_DIR, size=(32, 32)):
    images = load_images_from_folder(test_dir)
    names = sorted(
        f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")
    )

    if len(images) == 0:
        print(f"Nema slika u {test_dir}!")
        return [], []

    X = encode_images(images, size).reshape(-1, size[0], size[1], 3)

    probabilities = model.predict(X, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities[range(len(predicted)), predicted]

    results = [
        (name, LABEL_TO_CARD[label], float(conf))
        for name, label, conf in zip(names, predicted, confidence)
    ]
    return results, predicted


def save_results(results, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "predicted_card", "confidence"])
        writer.writerows(results)
    print(f"Rezultati sacuvani u {path}")


def main():
    model = load_model()

    print("=" * 60)
    print(f"PREDIKCIJA NA SLIKAMA IZ data/test/")
    print("=" * 60)

    results, _ = predict_directory(model)
    if not results:
        return

    print(f"{'slika':<12} {'predikcija':<20} {'verovatnoca':<12}")
    print("-" * 44)
    for name, card, conf in results:
        print(f"{name:<12} {card:<20} {conf:.2%}")

    save_results(results)


if __name__ == "__main__":
    main()