#!/usr/init/env python3
"""Module that creates a convolutional autoencoder."""
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (h, w, c).
        filters (list): Number of filters for each conv layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
    """
    # --- Encoder ---
    input_enc = K.Input(shape=input_dims)
    x = input_enc
    for f in filters:
        x = K.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Flatten and dense layer to match latent dimensions
    x = K.layers.Flatten()(x)
    latent = K.layers.Dense(units=latent_dims[0], activation='relu')(x)
    encoder = K.Model(inputs=input_enc, outputs=latent, name='encoder')

    # --- Decoder ---
    # To reconstruct the spatial dimensions prior to flattening/dense,
    # we need the shape before flattening. Let's compute it.
    # We can build a dummy model or track shapes, but standard practice
    # uses the feature map shape or a Dense layer to reshape back.
    # Let's determine the shape before Flatten in encoder:
    # Or alternatively, build encoder/decoder using functional layers:
    
    # Let's define decoder using Keras Functional API
    # First, let's find the shape of the tensor before Flatten in encoder
    # by instantiating an intermediate model or using shape inference.
    # Alternatively, specify latent input based on expected flattened shape:
    
    encoded_shape = encoder.output_shape[1:]
    
    # Let's construct decoder inputs and layers:
    input_dec = K.Input(shape=latent_dims)
    
    # We need to know what shape to dense-project to before reshaping.
    # Let's trace the downsampling shape through a dummy tensor:
    dummy_input = K.Input(shape=input_dims)
    d_x = dummy_input
    for f in filters:
        d_x = K.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(d_x)
        d_x = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(d_x)
    shape_before_flatten = K.backend.int_shape(d_x)[1:]

    # Now build the full decoder architecture:
    x_dec = K.layers.Dense(
        units=np.prod(shape_before_flatten),
        activation='relu'
    )(input_dec)
    x_dec = K.layers.Reshape(target_shape=shape_before_flatten)(x_dec)

    # Reversed filters for the decoder
    reversed_filters = filters[::-1]
    num_layers = len(reversed_filters)

    for i, f in enumerate(reversed_filters):
        if i == num_layers - 2:
            # Second to last convolution: valid padding, no upsampling
            x_dec = K.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                padding='valid',
                activation='relu'
            )(x_dec)
        elif i == num_layers - 1:
            # Last convolution: same number of filters as input channels,
            # sigmoid activation, no upsampling
            x_dec = K.layers.Conv2D(
                filters=input_dims[2],
                kernel_size=(3, 3),
                padding='same',
                activation='sigmoid'
            )(x_dec)
        else:
            # Standard decoder layer: conv with same padding, relu, upsampling
            x_dec = K.layers.Conv2D(
                filters=f,
                kernel_size=(3, 3),
                padding='same',
                activation='relu'
            )(x_dec)
            x_dec = K.layers.UpSampling2D(size=(2, 2))(x_dec)

    decoder = K.Model(inputs=input_dec, outputs=x_dec, name='decoder')

    # --- Full Autoencoder ---
    auto_output = decoder(encoder(input_enc))
    auto = K.Model(inputs=input_enc, outputs=auto_output, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
