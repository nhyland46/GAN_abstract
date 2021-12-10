import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gan import Discriminator, Generator
import os
from PIL import Image
import time
start = time.time()

def sample_noise(BATCH_SIZE, NOISE_DIM):
    """
    Returns a BATCH_SIZE x NOISE_DIM matrix of random uniform values.

    Inputs:
    -BATCH_SIZE
    -NOISE_DIM: desired dimension of noise vector
    """
    noise = tf.random.uniform((BATCH_SIZE, NOISE_DIM), minval = 0, maxval = 1)
    return noise

def train(discriminator, generator, d_optimizer, g_optimizer, batch_input):
    """
    Returns a tuple of (generator_loss, discriminator_loss)

    Inputs:
    -discriminator: a model of Discriminator classes
    -generator: a model of Generator class
    -optimizer: object of class tf.keras.optimizers
    -batch_input: a tensor of BATCH_SIZE x num_channels (3) x 128 x 128
    -latent: a seed tensor of BATCH_SIZE x NOISE_DIM to input into generator
    """
    # random noise fed into our generator
    BATCH_SIZE = np.shape(batch_input)[0]
    z = sample_noise(BATCH_SIZE, 1000)

    with tf.GradientTape(persistent=True) as tape: # persistent=True
      # generated images
      generated_images = generator(z)

      logits_real = discriminator(batch_input)

      # re-use discriminator weights on new inputs
      logits_fake = discriminator(generated_images)

      generator_loss = generator.loss_function(logits_fake)
      discriminator_loss = discriminator.loss_function(logits_fake, logits_real)

    # call optimize on the generator and the discriminator
    optimize(tape, generator, generator_loss, g_optimizer)
    optimize(tape, discriminator, discriminator_loss, d_optimizer)
    #generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
    #g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    #discriminator_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    #d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return generator_loss, discriminator_loss

# A bunch of utility functions
def show_images(images):
    """
    Shows a square grid of images using matplotlib.

    Inputs:
    -images : num_images x num_channels (3) x 128 x 128
    """
    #print('showing images...')
    #print('shape of images:',np.shape(images))
    #print('image pixel values:',images[0,0])
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

def optimize(tape: tf.GradientTape, model: tf.keras.Model, loss: tf.Tensor, optimizer) -> None:
    """
    Optimizes a model with respect to its loss.
    """
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def get_data(folder):
    print('extracting data...')
    image_list = []
    count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            #print('image',count)
            count+= 1
            img = Image.open(folder + '/' + filename)
            np_img = np.array(img)
            image_list.append(np_img)
        if count > 11000:
            break
    train_images = np.array(image_list)
    train_images = np.reshape(train_images, [train_images.shape[0], 3, 128,128]) #how should we be reshaping? this is from lab 8
    #print('train_images shape:', np.shape(train_images))
    train_images = train_images/255.0
    # show_images(train_images[0:4])
    #print('four images:',show_images(train_images[0:4]))
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
    folder = '/content/gdrive/My Drive/abstract1'
    train_dataset = get_data(folder)
    print('Data loaded')

    # Get an instance of VAE
    discriminator = Discriminator()
    generator = Generator()

    # HYPERPARAMS
    NUM_EPOCHS = 200
    BATCH_SIZE = 100
    NOISE_DIM = 1000
    EXAMPLES_EVERY = 20
    LOSS_EVERY = 20
    #N_DISCRIM = 5 #use this to train discriminator n times much as generator
    d_optimizer = tf.keras.optimizers.RMSprop(learning_rate = .00001)
    g_optimizer = tf.keras.optimizers.RMSprop(learning_rate = .00005)
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate = .25)

    epoch_n = 0
    d_losses = []
    g_losses = []
    for epoch in np.arange(NUM_EPOCHS):
        abstract = train_dataset.shuffle(BATCH_SIZE).batch(BATCH_SIZE)
        count = 0
        print('epoch#',epoch_n)
        for batch_input in abstract:
            # every show often, show a sample result
            print("")
            if count % EXAMPLES_EVERY == 0:
                latent = sample_noise(BATCH_SIZE, NOISE_DIM)
                samples = generator(latent)
                fig = show_images(samples[:16])
                plt.show()

            # run a batch of data through the network
            generator_loss, discriminator_loss = train(discriminator, generator, d_optimizer, g_optimizer, batch_input)
            d_losses.append(discriminator_loss)
            g_losses.append(discriminator_loss)
            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
            if count % LOSS_EVERY == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(count, discriminator_loss, generator_loss))
            print()
            count += 1
        epoch_n += 1
    # print("finished training at", time.time() - start)
    # Visualize results
    plt.plot(np.arange(2*epoch_n),d_losses)
    plt.show()
    plt.plot(np.arange(2*epoch_n),g_losses)
    plt.show()


if __name__ == "__main__":
    main()
