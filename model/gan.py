import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Reshape
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.constraints import Constraint
from keras import backend


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden_dim = 256
        self.const = ClipConstraint(0.01) # use this for conv layers

        self.discriminator = Sequential()
        ##TODO: change this to deconvolutions
        #use kernel_const=self.const for Conv layers
        #make sure final layer has linear activation function

        self.discriminator.add(Dense(self.hidden_dim))
        self.discriminator.add(LeakyReLU(alpha=0.01))
        self.discriminator.add(Dense(self.hidden_dim))
        self.discriminator.add(LeakyReLU(alpha=0.01))
        self.discriminator.add(Dense(1, activation='linear'))
        

    def call(self, images):
        """
        Runs the images through the discriminator network, returning a tensor of
        shape batch_size x 1 with the estimated LOGITS that the images are real.

        Inputs:
        - images : a tensor of batch_size x 3 x 128 x 128
        """
        return self.discriminator(images)

    def loss_function(self, logits_fake, logits_real):
        """
        Returns the discriminator's loss from fake and real images.

        Inputs:
        - logits_fake : a tensor of batch_size x 1 with the estimated logits of
        fake images from the generator
        - logits_real : a tensor of batch_size x 1 with the estimated logits of
        real images
        """
        #Old loss 
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
        # loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
        

        #Wasserstein loss
        #labels should be -1 for real and 1 for fake instead of 0 and 1
        #are these labels already taken into account?
        # labels_real = -tf.ones_like(logits_real)
        # labels_fake = tf.ones_like(logits_fake)

        #check sign, we want to maximize mean(real) - mean(fake), so loss is negative?
        # https://developers.google.com/machine-learning/gan/loss

        loss = -(tf.reduce_mean(logits_real) - tf.reduce_mean(logits_fake))
        return loss



class Generator(tf.keras.Model):

    """Model class for the generator"""
    def __init__(self):
        super(Generator, self).__init__()

        self.hidden_dim = 2048

        self.generator = Sequential()
         ##TODO: change this to convolutions

        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(self.hidden_dim))
        self.generator.add(LeakyReLU(alpha=0.01))
        self.generator.add(Dense(49152,activation = 'sigmoid')) # must be sigmoid because we want pixel values to be between 0 and 1

    def call(self, seed):
        """
        Returns the result of flowing the input noise tensor through the generator
        to get a matrix of batch_size x 3 x 128 x 128 images.

        Inputs:
        -seed: an input noise tensor of shape batch_size x noise_dim
        """
        return self.generator(seed)

    def loss_function(self, logits_fake):
        """
        Returns the generator's loss after running the generated images through the discriminator.

        Inputs:
        - logits_fake : a tensor of batch_size x 1 with the estimated logits of
        fake generated images
        """
        #Old loss
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(logits_fake),logits = logits_fake,name = 'generator_loss'))
        
        #Wasserstein loss
        #check sign, we want to maximimize mean(logits_fake), so loss should be negative of that?
        # https://developers.google.com/machine-learning/gan/loss
        return -tf.reduce_mean(logits_fake)

class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
