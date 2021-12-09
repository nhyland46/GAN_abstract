import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gan import Discriminator, Generator
import os
from PIL import Image
import time
start = time.time()

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

def sample_noise(BATCH_SIZE, NOISE_DIM):
    """
    Returns a BATCH_SIZE x NOISE_DIM matrix of random uniform values.

    Inputs:
    -BATCH_SIZE
    -NOISE_DIM: desired dimension of noise vector
    """
    return tf.random.uniform((BATCH_SIZE, NOISE_DIM), minval = -1, maxval = 1)

def preprocess_image(x):
    """
    COMMENTS
    """
    return 2 * x - 1.0

def train(discriminator, generator, optimizer, batch_input, latent):
    """
    Returns a tuple of (generator_loss, discriminator_loss)

    Inputs:
    -discriminator: a model of Discriminator classes
    -generator: a model of Generator class
    -optimizer: object of class tf.keras.optimizers
    -batch_input: a tensor of BATCH_SIZE x num_channels (3) x 128 x 128
    -latent: a seed tensor of BATCH_SIZE x NOISE_DIM to input into generator
    """
    ##TODO: change to train discriminator more than generator (5x more?)

    # random noise fed into our generator
    BATCH_SIZE = np.shape(batch_input)[0]
    z = latent

    with tf.GradientTape(persistent=True) as tape: # persistent=True
      # generated images
      generated_images = generator(z)

      logits_real = discriminator(preprocess_image(batch_input))

      # re-use discriminator weights on new inputs
      logits_fake = discriminator(generated_images)

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
    Shows a square grid of images using matplotlib.

    Inputs:
    -images : num_images x num_channels (3) x 128 x 128
    """
    print('showing images...')
    print('shape of images:',np.shape(images))
    print('image pixel values:',images[0,0])
    images = np.reshape(images, [images.shape[0], 3, 128, 128])  # images reshape to (BATCH_SIZE, D)
    #print('images shape:',np.shape(images))
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    dim = 128 #height or width of image

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
        plt.imshow(img.reshape([dim,dim,3]))
    return

def get_data(folder):
    print('extracting data...')
    image_list = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img = Image.open(folder + '/' + filename)
            np_img = np.array(img)
            image_list.append(np_img)
    train_images = np.array(image_list)
    #print('train_images shape:', np.shape(train_images))
    train_images = np.reshape(train_images, [train_images.shape[0], 3, 128,128]) #how should we be reshaping? this is from lab 8
    print('train_images shape:', np.shape(train_images))
    train_images = train_images/255.0
    # show_images(train_images[0:4])
    print('four images:',show_images(train_images[0:4]))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    return train_dataset


def main():
    # Load Abstract datasett
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
    NUM_EPOCHS = 1
    BATCH_SIZE = 100
    NOISE_DIM = 50
    EXAMPLES_EVERY = 200
    LOSS_EVERY = 50
    N_DISCRIM = 5 #use this to train discriminator n times much as generator
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = .00005)

    abstract = train_dataset.repeat(NUM_EPOCHS).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
    for epoch in np.arange(NUM_EPOCHS):
        count = 0
        for batch_input in abstract:
            print("batch", count)
            print('batch shape', np.shape(batch_input))
            latent = sample_noise(BATCH_SIZE, NOISE_DIM)
            # every show often, show a sample result
            print("")
            if count % EXAMPLES_EVERY == 0:
                samples = generator(latent)
                fig = show_images(samples[:16])
                plt.show()

            # run a batch of data through the network
            generator_loss, discriminator_loss = train(discriminator, generator, optimizer, batch_input, latent)

            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
            if count % LOSS_EVERY == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(count, discriminator_loss, generator_loss))
            print()
            count += 1
    print("finished training at", time.time() - start)
    # Visualize results


if __name__ == "__main__":
    main()
