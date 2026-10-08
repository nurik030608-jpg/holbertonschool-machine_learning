#!/usr/bin/env python3
"""Module that optimizes a Keras neural network using GPyOpt."""

import os
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist


# 1. Load and preprocess dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = (x_train.astype('float32') / 255.0).reshape(-1, 28 * 28)
x_val = (x_val.astype('float32') / 255.0).reshape(-1, 28 * 28)

# Directory to save model checkpoints
os.makedirs("checkpoints", exist_ok=True)


# 2. Objective Function for GPyOpt
def fit_and_evaluate(domain_params):
    """Fits model with hyperparameter configuration and returns validation loss.

    Args:
        domain_params (numpy.ndarray): Array containing domain hyperparameter
            values: [[learning_rate, num_units, dropout_rate, l2_weight,
            batch_size]]

    Returns:
        float: Satisficing metric (Validation Loss to be minimized).
    """
    params = domain_params[0]
    lr = float(params[0])
    num_units = int(params[1])
    dropout_rate = float(params[2])
    l2_weight = float(params[3])
    batch_size = int(params[4])

    # Build Keras Model
    model = models.Sequential([
        layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight),
            input_shape=(28 * 28,)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(
            num_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_weight)
        ),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Checkpoint filename encoding hyperparameter configuration
    ckpt_filename = (
        f"checkpoints/best_model_lr{lr:.5f}_units{num_units}_"
        f"drop{dropout_rate:.3f}_l2{l2_weight:.5f}_bs{batch_size}.h5"
    )

    checkpoint_cb = ModelCheckpoint(
        filepath=ckpt_filename,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=0
    )

    # Return minimum validation loss achieved (satisficing metric)
    best_val_loss = float(np.min(history.history['val_loss']))
    return best_val_loss


# 3. Define Hyperparameter Search Domain
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128, 256)}
]


# 4. Perform Bayesian Optimization
def main():
    """Runs Bayesian Optimization using GPyOpt and logs results."""
    np.random.seed(42)
    tf.random.set_seed(42)

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=fit_and_evaluate,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        exact_feval=True,
        maximize=False
    )

    # Run optimization for maximum 30 iterations
    max_iter = 30
    optimizer.run_optimization(max_iter=max_iter)

    # Plot convergence
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')
    plt.close()

    # Log best parameters and loss report
    best_params = optimizer.x_
