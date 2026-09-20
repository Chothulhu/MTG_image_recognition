import tensorflow as tf


def build_cnn(input_shape=(32, 32, 3), num_classes=5):
    """CNN za klasifikaciju slika kartica.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Prvi konv. sloj sa 32 filtera 3x3
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Drugi konv. sloj sa 64 filtera 3x3
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Prretvaramo 3D u 1D
        tf.keras.layers.Flatten(),

        # Potpuno povezani sloj sa 128 neurona
        tf.keras.layers.Dense(128, activation="relu"),

        # 30% nasumicno iskljucenih neurona pri svakom koraku da bi sprecili overfit
        tf.keras.layers.Dropout(0.3),

        # num_classes neurona, softmax verovatnoce za svaku klasu
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    return model