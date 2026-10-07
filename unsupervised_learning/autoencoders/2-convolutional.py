import tensorflow as tf


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims: tuple of integers, dimensions of the model input
        filters: list of integers, number of filters for each convolutional layer in the encoder
        latent_dims: tuple of integers, dimensions of the latent space representation

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model compiled with adam and binary_crossentropy
    """
    # -------------------
    # ENCODER
    # -------------------
    encoder_inputs = tf.keras.Input(shape=input_dims)
    x = encoder_inputs

    # Проходим по фильтрам для энкодера
    for f in filters:
        x = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder_outputs = x  # Должно соответствовать latent_dims
    encoder = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_outputs, name='encoder')

    # -------------------
    # DECODER
    # -------------------
    decoder_inputs = tf.keras.Input(shape=latent_dims)
    x = decoder_inputs

    # Фильтры для декодера меняются на обратные
    reversed_filters = filters[::-1]

    # Все свертки кроме последних двух
    for f in reversed_filters[:-1]:
        x = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    # Предпоследняя свертка: padding='valid' с UpSampling2D
    x = tf.keras.layers.Conv2D(
        filters=reversed_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    # Последняя свертка: фильтры = количество каналов в input_dims, activation='sigmoid', без UpSampling
    num_channels = input_dims[-1]
    decoder_outputs = tf.keras.layers.Conv2D(
        filters=num_channels,
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_outputs, name='decoder')

    # -------------------
    # AUTOENCODER
    # -------------------
    auto_inputs = encoder_inputs
    encoded_repr = encoder(auto_inputs)
    reconstructed = decoder(encoded_repr)

    auto = tf.keras.Model(inputs=auto_inputs, outputs=reconstructed, name='autoencoder')

    # Компиляция автоэнкодера с оптимизатором Adam и бинарной кросс-энтропией
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
