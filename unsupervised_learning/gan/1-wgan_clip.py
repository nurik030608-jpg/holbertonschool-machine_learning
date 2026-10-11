#!/usr/bin/env python3
"""Module that defines the WGAN_clip class with weight clipping."""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Represents a Wasserstein GAN with weight clipping."""

    def __init__(
        self, generator, discriminator, latent_generator,
        real_examples, batch_size=200, disc_iter=2,
        learning_rate=0.005
    ):
        """Initialize the WGAN_clip instance.

        Args:
            generator (keras.Model): Generator network.
            discriminator (keras.Model): Discriminator/Critic network.
            latent_generator (callable): Function to generate latent vectors.
            real_examples (tf.Tensor): Dataset of real examples.
            batch_size (int, optional): Size of training batches.
                Defaults to 200.
            disc_iter (int, optional): Discriminator steps per generator step.
                Defaults to 2.
            learning_rate (float, optional): Learning rate for Adam.
                Defaults to 0.005.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Define generator loss (Wasserstein: -mean(discriminator(fake)))
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Define discriminator loss (Wasserstein: mean(fake) - mean(real))
        self.discriminator.loss = (
            lambda real_output, fake_output: tf.math.reduce_mean(fake_output)
            - tf.math.reduce_mean(real_output)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples using the generator network."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Sample a batch of real examples from the dataset."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """Execute a single training step for WGAN with weight clipping."""
        # Train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                disc_real = self.discriminator(real_sample, training=True)
                disc_fake = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(disc_real, disc_fake)

            gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables)
            )

            # Clip the weights of the discriminator between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # Train the generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            disc_fake = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(disc_fake)

        gradients = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
