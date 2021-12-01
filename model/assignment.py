import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from gan import Discriminator, Generator
import os
from PIL import Image


# will we need this?
def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset 
    Returns:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when 
    the ground truth label for image i is j, and targets[i, :j] & 
    targets[i, j + 1:] are equal to 0
    """
    targets = np.zeros((labels.shape[0], class_size))
    for i, label in enumerate(labels):
        targets[i, label] = 1
    targets = tf.convert_to_tensor(targets)
    targets = tf.cast(targets, tf.float32)
    return targets

def sample_noise(batch_size, dimension):
    """
    COMMENTS
    """
    return tf.random.uniform((batch_size, dimension), minval = -1, maxval = 1)

def preprocess_image(x):
    """
    COMMENTS
    """
    return 2 * x - 1.0

def train(discriminator, generator, batch_input, batch_size, noise_dimension):
    """
    COMMENT
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate = .001)

    # random noise fed into our generator
    z = sample_noise(batch_size, noise_dimension)

    with tf.GradientTape() as tape: # persistent=True
      # generated images
      generated_images = generator(z)

      # scale images to be -1 to 1
      logits_real = discriminator(preprocess_image(batch_input))
      # re-use discriminator weights on new inputs
      logits_fake = discriminator(generated_images)

      generator_loss = generator.loss_function(logits_fake, logits_real)
      discriminator_loss = discriminator.loss_function(logits_fake, logits_real)
      
    # call optimize on the generator and the discriminator
    generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))  
    
    discriminator_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))  
   
    return generator_loss, discriminator_loss

# A bunch of utility functions
def show_images(images):
    """
    COMMENT
    """
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def get_data(folder): 
    image_list = []
    for filename in os.listdir(folder): 
        if filename.endswith('.jpg'):
            img = Image.open(folder + '/' + filename)
            np_img = np.array(img)
            image_list.append(np_img)
    train_images = np.array(image_list)
    train_images = np.reshape(train_images, [train_images.shape[0], -1]) #how should we be reshaping? this is from lab 8
    train_images = train_images/255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    return train_dataset


def main():
    # Load Abstract dataset

    #Default 
    # folder = None

    #Nick
    folder = '/Users/nickhyland/Desktop/abstract128'

    #Olivia 
    # folder = ?

    #Kevin 
    # folder = ?

    train_dataset = get_data(folder)
    print('Preprocessing Done.')
    # Get an instance of VAE
    discriminator = Discriminator()
    generator = Generator()

    # Train VAE
    num_epochs = 10
    batch_size = 128
    noise_dim = 4 #need to change this value. somehow it is receiving (128, 49152) for loss calculation. Loss calculation different?

    count = 0
    abstract = train_dataset.repeat(num_epochs).shuffle(batch_size).batch(batch_size)
    for batch_input in abstract:
        # every show often, show a sample result
        if count % 250 == 0:
            samples = generator(sample_noise(batch_size, noise_dim))
            fig = show_images(samples[:16])
            plt.show()
            print()
        # run a batch of data through the network
        print(batch_input.shape)
        generator_loss, discriminator_loss = train(generator, discriminator, batch_input, batch_size, noise_dim)

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if count % 50 == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(count, discriminator_loss, generator_loss))
        count += 1

    # Visualize results
    # show_images()


if __name__ == "__main__":
    main()
