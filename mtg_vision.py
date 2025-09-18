from PIL import Image, ImageEnhance
import random
import os
import numpy as np


# flip, rotacija, osvetljenje, kontrast. dodati jos?
def augment_image(img):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.rotate(random.randint(-15, 15))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


def augment_image_x_times(img, x):
    augmented_images = []

    # za svaku sliku x augmentovanih
    for _ in range(x):
        augmented_images.append(augment_image(img.copy()))
    return augmented_images


def load_images_from_folder(folder, label, size=(32, 32)):
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert("RGB")
            augmented_images = augment_image_x_times(img, 10)
            save_images(
                np.asarray(augmented_images),
                "data/augmented/test/augmented_images",
                filename,
            )
            for img in augmented_images:
                img = img.resize(size)
                img_array = np.array(img).astype(np.float32) / 255.0
                # print(img_array.flatten())
                images.append(img_array.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)


def save_images(images, output_folder, name=""):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, img_array in enumerate(images):
        img = Image.fromarray((img_array * 255).astype("uint8"))
        img.save(os.path.join(output_folder, f"augmented_{name}_{i}.jpg"))


angelic_renewal_images, angelic_renewal_labels = load_images_from_folder(
    "data/base/angelic_renewal", 0
)  # 0 je labela za sliku "angelic_renewal"
save_images(angelic_renewal_images, "data/augmented/augmented_angelic_renewal")

ironshell_beetle_images, ironshell_beetle_labels = load_images_from_folder(
    "data/base/ironshell_beetle", 1
)  # 1 je labela za sliku "ironshell_beetle"
save_images(
    ironshell_beetle_images, "data/augmented/augmented_ironshell_beetle"
)

jade_avenger_images, jade_avenger_labels = load_images_from_folder(
    "data/base/jade_avenger", 2
)  # 2 je labela za sliku "jade_avenger"
save_images(jade_avenger_images, "data/augmented/augmented_jade_avenger")

soulherder_images, soulherder_labels = load_images_from_folder(
    "data/base/soulherder", 3
)  # 3 je labela za sliku "soulherder"
save_images(soulherder_images, "data/augmented/augmented_soulherder")

thallid_images, thallid_labels = load_images_from_folder(
    "data/base/thallid", 4
)  # 4 je labela za sliku "thallid"
save_images(thallid_images, "data/augmented/augmented_thallid")
