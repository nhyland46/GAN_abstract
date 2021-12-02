import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Reshape
from tensorflow.math import exp, sqrt, square


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden_dim = 256

        self.discriminator = Sequential()
        self.discriminator.add(Dense(self.hidden_dim))
        self.discriminator.add(LeakyReLU(alpha=0.01))
        self.discriminator.add(Dense(self.hidden_dim))
        self.discriminator.add(LeakyReLU(alpha=0.01))
        self.discriminator.add(Dense(1))


    def call(self, x):
        """
        COMMENTS
        """
        return self.discriminator(x)

    def loss_function(self, logits_fake, logits_real):
        """
        COMMENTS
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
        return loss



class Generator(tf.keras.Model):

    """Model class for the generator"""
    def __init__(self):
        super(Generator, self).__init__()

        self.hidden_dim = 2048

        self.generator = Sequential()
        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(49152,activation = 'sigmoid')) # must be sigmoid because we want pixel values to be between 0 and 1


    def call(self, x):
        """
        COMMENTS
        """
        print('running generator')
        output = self.generator(x)
        print(np.shape(output))
        return(output)

    def loss_function(self, logits_fake):
        """
        COMMENTS
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake),logits = logits_fake,name = 'generator_loss'))
        return loss
