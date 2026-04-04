#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture
    """
    # Определение инициализатора HeNormal с фиксированным зерном (seed=0)
    initializer = K.initializers.HeNormal(seed=0)

    # 1. Convolutional layer: 6 kernels (5x5), same padding, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # 2. Max pooling layer: kernels (2x2), strides (2x2)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # 3. Convolutional layer: 16 kernels (5x5), valid padding, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # 4. Max pooling layer: kernels (2x2), strides (2x2)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Перед Fully Connected слоями нужно преобразовать 2D в 1D
    flatten = K.layers.Flatten()(pool2)

    # 5. Fully connected layer: 120 nodes, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # 6. Fully connected layer: 84 nodes, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # 7. Softmax output layer: 10 nodes
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    # Создание модели
    model = K.Model(inputs=X, outputs=output)

    # Компиляция модели с оптимизатором Adam и метрикой accuracy
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
