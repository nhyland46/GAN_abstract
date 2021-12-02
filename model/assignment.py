import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
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

def train(discriminator, generator, batch_input, latent):
    """
    Returns:
    -
    """
    print('starting training')
    optimizer = tf.keras.optimizers.Adam(learning_rate = .001)

    # random noise fed into our generator
    batch_size = np.shape(batch_input)[0]
    z = latent
    print("latent noise shape:", np.shape(z))
    with tf.GradientTape() as tape: # persistent=True
      # generated images
      generated_images = generator(z) #this is not working, the generated images are batch_size x 1 but they should be batch_size x 49152
      #print("real_images shape:", np.shape(batch_input))
      print("generated_images shape:", np.shape(generated_images))
      logits_real = discriminator(preprocess_image(batch_input))
      print("real images logits shape:", np.shape(logits_real))
      # re-use discriminator weights on new inputs
      logits_fake = discriminator(generated_images)
      print("fake images logits shape:", np.shape(logits_fake))
      generator_loss = generator.loss_function(logits_fake)
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
    Shows example images.
    """
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    #print('images shape:',np.shape(images))
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1]/3)))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        #print('16 pixels of one image:', img[:16])
        plt.imshow(img.reshape([sqrtimg,sqrtimg,3]))
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
    #print('train_images shape:', np.shape(train_images))
    train_images = train_images/255.0
    print('one image:',train_images[0])
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    return train_dataset


def main():
    # Load Abstract dataset

    #Default
    # folder = None

    #Nick
    #folder = '/Users/nickhyland/Desktop/abstract128'

    #Olivia
    # folder = ?

    #Kevin
    folder = '/Users/kevinma/Downloads/abstract128'
    train_dataset = get_data(folder)
    print('Data loaded')
    # Get an instance of VAE
    discriminator = Discriminator()
    generator = Generator()

    # HYPERPARAMS
    num_epochs = 10
    batch_size = 128
    noise_dim = 50 #need to change this value. somehow it is receiving (128, 49152) for loss calculation. Loss calculation different?
    examples_every = 250
    loss_every = 50

    count = 0
    abstract = train_dataset.repeat(num_epochs).shuffle(batch_size).batch(batch_size)
    for batch_input in abstract:
        # every show often, show a sample result
        latent = sample_noise(batch_size, noise_dim)
        # if count % examples_every == 0:
        #     #print("batch_input size:", np.shape(batch_input))
        #     #print('is any pixel negative?', any(ele < 0 for ele in batch_input[0]))
        #     samples = generator(latent)
        #     print('samples shape:',np.shape(samples))
        #     fig = show_images(samples[:16])
        #     plt.show()
        #     print()
        # run a batch of data through the network
        #print(np.shape(batch_input))
        generator_loss, discriminator_loss = train(generator, discriminator, batch_input, latent)

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if count % loss_every == 0:
            print('Iter: {}, D: {:.4}, G:{:.4}'.format(count, discriminator_loss, generator_loss))
        count += 1

    # Visualize results
    # show_images()


if __name__ == "__main__":
    main()
