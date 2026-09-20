import os

import numpy as np
from PIL import Image


def load_images_from_folder(folder):
    """Ucitava sve .jpg slike iz foldera kao PIL Image objekte (RGB)."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            images.append(Image.open(path).convert("RGB"))
    return images


def encode_image(img, size=(32, 32)):
    """Resize, normalizacija na [0, 1] i spljostavanje u 1D niz."""
    # LANCZOS daje najostriju sliku pri smanjivanju visoke rezolucije
    img = img.resize(size, Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array.flatten()


def encode_images(images, size=(32, 32)):
    """Enkoduje listu PIL slika u 2D numpy niz (N, size*size*3)."""
    return np.array([encode_image(img, size) for img in images])


def save_pil_images(images, output_folder, name=""):
    """Cuvanje PIL slika u punoj rezoluciji."""
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(output_folder, f"{name}_{i}.jpg"))


def save_encoded_images(images, output_folder, name="", size=(32, 32)):
    """Cuvanje enkodovanih (32x32) nizova kao slika, radi vizuelnog pregleda."""
    os.makedirs(output_folder, exist_ok=True)
    rows, cols = size
    for i, arr in enumerate(images):
        img_array = (arr.reshape(rows, cols, 3) * 255).astype("uint8")
        Image.fromarray(img_array).save(os.path.join(output_folder, f"{name}_{i}.jpg"))