#!/usr/bin/env python3
"""
Модуль для создания вариационного автоэнкодера (VAE)
"""
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer
                       in the encoder
        latent_dims: integer, dimensions of the latent space representation

    Returns:
        encoder: the encoder model (outputs latent representation, mean, log variance)
        decoder: the decoder model
        auto: the full autoencoder model
    """
    # -------------------
    # SAMPLING LAYER
    # -------------------
    class Sampling(tf.keras.layers.Layer):
        """Репараметризация (Reparameterization trick)"""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # -------------------
    # ENCODER
    # -------------------
    encoder_inputs = tf.keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(units=nodes, activation='relu')(x)

    # Слои для среднего значение (mean) и логорифма вариации (log variance) с activation=None
    z_mean = tf.keras.layers.Dense(units=latent_dims, activation=None, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(units=latent_dims, activation=None, name='z_log_var')(x)

    # Семплирование скрытого представления
    z = Sampling()([z_mean, z_log_var])

    # Энкодер возвращает: latent representation (z), mean (z_mean), log variance (z_log_var)
    encoder = tf.keras.Model(
        inputs=encoder_inputs,
        outputs=[z, z_mean, z_log_var],
        name='encoder'
    )

    # -------------------
    # DECODER
    # -------------------
    decoder_inputs = tf.keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    # Слои декодера выстраиваются в обратном порядке
    for nodes in reversed(hidden_layers):
        x = tf.keras.layers.Dense(units=nodes, activation='relu')(x)

    # Последний слой с активацией sigmoid
    decoder_outputs = tf.keras.layers.Dense(units=input_dims, activation='sigmoid')(x)

    decoder = tf.keras.Model(inputs=decoder_inputs, outputs=decoder_outputs, name='decoder')

    # -------------------
    # AUTOENCODER (VAE)
    # -------------------
    auto_inputs = encoder_inputs
    # Получаем z из энкодера (первый элемент кортежа outputs)
    latent_repr, _, _ = encoder(auto_inputs)
    reconstructed = decoder(latent_repr)

    auto = tf.keras.Model(inputs=auto_inputs, outputs=reconstructed, name='auto')

    # Компиляция с Adam и binary_crossentropy
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
