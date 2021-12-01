import tensorflow as tf
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

    def loss_function(self, predictions, labels):
        """
        COMMENTS
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions), logits=predictions))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(labels), logits=labels))
        return loss
    


class Generator(tf.keras.Model):
  
    """Model class for the generator"""
    def __init__(self):
        super(Generator, self).__init__()
        
        self.hidden_dim = 1024

        self.generator = Sequential()
        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(784,activation = tf.nn.tanh))

    
    def call(self, x):
        """
        COMMENTS
        """
        return self.generator(x)
        
    def loss_function(self, predictions, labels):
        """
        COMMENTS
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(predictions),logits = predictions,name = 'generator_loss'))
        return loss