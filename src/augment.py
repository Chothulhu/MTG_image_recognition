"""Augmentacija slika."""

import random

from PIL import Image, ImageEnhance


def _rotate_and_crop(img, angle):
    """Rotira sliku i vraca je na originalnu velicinu centriranim crop-om.

    Za razliku od obicnog rotate (koji crne uglove popunjava crnom ili
    bojom ivice), ovde nema sinteticke popune — u kadru ostaju samo
    pravi pikseli karte, kao na fotografiji.
    """
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    width, height = img.size
    left = (rotated.width - width) // 2
    top = (rotated.height - height) // 2
    return rotated.crop((left, top, left + width, top + height))


def augment_image(img):
    """Nasumicno menja sliku: flip, rotacija, osvetljenje, kontrast, zasicenje."""
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.5:
        img = _rotate_and_crop(img, random.randint(-15, 15))

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    return img


def augment_image_x_times(img, x):
    """Vraca listu od x augmentovanih kopija slike."""
    augmented_images = []
    for _ in range(x):
        augmented_images.append(augment_image(img.copy()))
    return augmented_images