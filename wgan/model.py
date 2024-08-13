# import statements
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import datetime
import shutil
import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Flatten, Dropout, Reshape, ReLU, PReLU, LeakyReLU
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Conv2DTranspose

from IPython import display

def initialize_generator(input_dims = (64, 64, 3), 
                         max_filters = 1024, 
                         kernel_size = 5, 
                         noise_dim = 128,
                         leaky_relu_slope = None,
                         model_name = 'generator'):
    """
    Function to initialize the generator architecture according to specified parameters.
    
    :param tuple input_dims: Input dimensions for images (img_height, img_width, color_channels).
    :param int max_filters: Maximum number of filters for the lowest layer in the generator.
    :param int kernel_size: Kernel size used.
    :param int noise_dim: Number of elements in the latent space.
    :param float leaky_relu_slope: negative slope of leaky ReLu activation. If None, the model uses Parametric ReLU (PReLU) 
                                   and attempts to arrive at the slope through learnable parameters. Defaults to None.
    :param str model_name: Name of model.
    
    :return: tf.keras.Sequential generator: generator architecture.
    """

    # do a little math to determine the number of upsamplings required in both the height and width
    # dimensions to arrive at the input dimensions
    n_upsamples_height = (tf.math.log(float(input_dims[0]))/tf.math.log(2.))-2
    n_upsamples_width = (tf.math.log(float(input_dims[1]))/tf.math.log(2.))-2
    
    # make sure the number of upsamplings along the height & width dimensions correspond
    assert n_upsamples_height == n_upsamples_width

    f = max_filters # filter tracker
    layer_list = [] # layer list

    # populate layer list based on parameters
    layer_list.append(Input(shape=(noise_dim,), name = "Input"))

    layer_list.append(Dense(4 * 4 * max_filters, name = "FC"))
    layer_list.append(BatchNormalization(name = "BN"))
    
    if leaky_relu_slope == None:
        layer_list.append(PReLU(name = "PReLu"))
    else:
        layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu"))
    
    layer_list.append(Reshape((4, 4, -1), name = "Reshape"))

    layer_list.append(Conv2DTranspose(f, kernel_size = kernel_size, strides = 1, padding = "same", kernel_initializer = "glorot_normal", name = "TConv2D_1"))
    layer_list.append(BatchNormalization(name = "BN_1"))
    if leaky_relu_slope == None:
        layer_list.append(PReLU(name = "PReLu_1"))
    else:
        layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu_1"))
        pass
    
    f = int(f/2)

    # add the transposed convolution blocks with stride 2 (ensuring upsampling behaviour)
    for i in range(int(n_upsamples_height)):
        layer_list.append(Conv2DTranspose(f, kernel_size = kernel_size, strides = 2, padding = "same", 
                                          kernel_initializer = "glorot_normal", name = "TConv2D_"+str(i+2)))
        layer_list.append(BatchNormalization(name = "BN_"+str(i+2)))
        if leaky_relu_slope == None:
            layer_list.append(PReLU(name = "PReLu_"+str(i+2)))
        else:
            layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu_"+str(i+2)))
        f = int(f/2)
        pass

    # final layer brings the number of channels back down to 3 (RGB)
    layer_list.append(Conv2D(3, kernel_size = kernel_size, padding = "same", 
                             activation = "tanh", kernel_initializer = "glorot_normal", name = "TConv2D_out"))

    generator = Sequential(layer_list, name = model_name)

    return generator

def initialize_discriminator(input_dims = (64, 64, 3), 
                             max_filters = 1024, 
                             kernel_size = 5, 
                             noise_dim = 128,
                             leaky_relu_slope = 0.2,
                             model_name = 'discriminator'):
    """
    Function to initialize the discriminator architecture according to specified parameters.
    
    :param tuple input_dims: Input dimensions for images (img_height, img_width, color_channels).
    :param int max_filters: Maximum number of filters for the top layer in the discriminator.
    :param int kernel_size: Kernel size used.
    :param int noise_dim: Number of elements in the latent space.
    :param float leaky_relu_slope: negative slope of leaky ReLu activation. If None, the model uses Parametric ReLU (PReLU) 
                                   and attempts to arrive at the slope through learnable parameters.
    :param str model_name: Name of model.
    
    :return: tf.keras.Sequential discriminator: discriminator architecture.
    """

    # do a little math to determine the number of upsamplings required in both the height and width
    # dimensions to arrive from the input dimensions to the desired feature map
    n_upsamples_height = (tf.math.log(float(input_dims[0]))/tf.math.log(2.))-1
    n_upsamples_width = (tf.math.log(float(input_dims[1]))/tf.math.log(2.))-1
    assert n_upsamples_height == n_upsamples_width 

    # extract minimum number of filters
    f = max_filters
    for i in range(int(n_upsamples_height-1)):
        f = int(f/2)
        pass

    layer_list = [] # layer list

    # populate layer list based on parameters
    layer_list.append(Input(shape=input_dims, name = "Input"))

    for i in range(int(n_upsamples_height)):
        layer_list.append(Conv2D(f, kernel_size = kernel_size, strides = 2, padding = "same", kernel_initializer = "glorot_normal", name = "Conv2D_"+str(i+1)))
        layer_list.append(BatchNormalization(name = "BN_"+str(i+1)))
        if leaky_relu_slope == None:
            layer_list.append(PReLU(name = "PReLu_"+str(i+1)))
        else:
            layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu_"+str(i+1)))
        f = int(f*2)
        pass

    layer_list.append(Flatten())
    layer_list.append(Dropout(0.3))
    layer_list.append(Dense(1))
    
    discriminator = Sequential(layer_list, name = model_name)

    return discriminator

class WGAN(Model):
    
    ####################################################################################
    # this code is based on the code from: https://keras.io/examples/generative/wgan_gp/
    ####################################################################################

    def __init__(self, discriminator, generator, noise_dim, k = 3, gp_weight = 10.0, model_name = "WGAN"):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.noise_dim = noise_dim
        self.k = k
        self.gp_weight = gp_weight
        self._name = model_name
        
        # makes sure that the image dimensions correspond with the discriminator and generator
        assert(self.generator.layers[-1].output.shape[1:] == self.discriminator.input.shape[1:])
        pass
    
    def compile(self, dis_optimizer, gen_optimizer):
        super().compile()
        self.dis_optimizer = dis_optimizer
        self.gen_optimizer = gen_optimizer
        pass
        
    def gradient_penalty(self, batch_size, real_images, fake_images):
        
        # one uniformly random number generated for each image in the batch 
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0) 
        
        # interpolated images are real images randomly adjusted for the distance to the fake images
        interpolated = tf.add(real_images, tf.multiply(alpha, tf.subtract(fake_images, real_images))) 

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated) # makes sures the interpolated image is included in the graph
            pred = self.discriminator(interpolated, training = True) # run interpolated image through discriminator
            pass

        # calculate the gradient penalty
        grads = gp_tape.gradient(pred, [interpolated])[0] # get the gradients w.r.t to this interpolated image.        
        norm = tf.sqrt(tf.add(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]), 1e-12)) # calculate norm of gradients
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp

    def train_step(self, real_images):
        
        # extract batch size
        batch_size = tf.shape(real_images)[0]


        ################## TRAIN DISCRIMINATOR ##################
        for i in range(self.k): # we train the discriminator k times for every time we train the generator
            
            # generate random noise to feed to the generator for training
            noise = tf.random.normal(shape = (batch_size, self.noise_dim))
            
            with tf.GradientTape() as tape:
                
                # generate images from noise
                generated_images = self.generator(noise, training = True)
                # run generated images through discriminator
                y_fake = self.discriminator(generated_images, training = True)
                # run the real images through the discriminator
                y_true = self.discriminator(real_images, training = True)

                # discriminator loss is essentially the distance between the actual and generated distributions
                # having the discriminator loss approach zero too fast --> discriminator becomes to good at distinguishing real from fake
                # ideally we would like to see a gradual/overall trend towards a larger negative discriminator loss (meaning it classifies more images as "real")
                discriminator_loss = tf.subtract(tf.reduce_mean(y_fake), tf.reduce_mean(y_true))
                
                # gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, generated_images)
                
                # add gradient penalty to the discriminator loss
                discriminator_loss = discriminator_loss + tf.multiply(gp, self.gp_weight)

            # get gradients w.r.t the discriminator loss
            discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            # update discriminator weights
            self.dis_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))


        ################## TRAIN GENERATOR ##################
        noise = tf.random.normal(shape=(batch_size, self.noise_dim))
        
        with tf.GradientTape() as tape:
            # generate images from noise
            generated_images = self.generator(noise, training = True)
            # run generated image through discriminator
            y_fake = self.discriminator(generated_images, training = True)
            # calculate the generator loss
            # whereas we would like to see discriminator loss approaching zero, we would like to see generator loss
            # grow larger in either direction. The discriminator loss function ensures that real en fake image predictions are "pulled apart".
            generator_loss = -tf.reduce_mean(y_fake)
            pass

        # get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        # update generator weights
        self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        
        return {"dis_loss": discriminator_loss, "gen_loss": generator_loss}

def generate_and_save_sample(imgs, n, epoch, folder, save = True):
    
    # clear existing console output
    display.clear_output(wait = True)

    # construct image array
    for i in range(int(n**2)):
        plt.subplot(n, n, i+1)
        plt.imshow(imgs[i,]/255.)
        plt.axis('off')
        pass

    # save image to designated folder
    plt.suptitle('WGAN - epoch #' + str(int(epoch)), fontsize = 10)
    if save:
        plt.savefig(os.getcwd() + '/' + folder + '/images_at_epoch_{:04d}.png'.format(epoch))
        pass
    plt.show()
    pass

class monitor_wgan_performance(Callback):
    
    def __init__(self, noise, folder = 'training_progress'):
        
        # pass in consistent latent vector of noise to accuractely asses 
        # the progress made by the generator on the "same" input
        self.noise = noise
        self.folder = folder
        self.n = int(np.floor(np.sqrt(noise.shape[0])))
        self.offset = 0
        pass
        
    def on_train_begin(self, logs = None):
        
        # create a sub-directory for the model (based on its name)
        self.folder = self.folder + '/' + self.model.name
        
        # if training progress folder does not exist...
        if not os.path.exists(os.getcwd() + '/' + self.folder):
            os.makedirs(os.getcwd() + '/' + self.folder) # create one...
            pass
        else: # if a training progress folder does already exist...
            #shutil.rmtree(os.getcwd() + '/' + self.folder) # remove it and...
            #os.makedirs(os.getcwd() + '/' + self.folder) # create a new one.
            self.offset = len(os.listdir(os.getcwd() + '/' + self.folder)) # continue keeping track of training progress
            pass
        
        # run noise through current iteration of generator
        generated_images = self.model.generator(self.noise)
        generated_images = (generated_images * 127.5) + 127.5 # scale predicitons from [-1,1] to [0,255]
        generated_images.numpy()
        pass
        
        # generate images from random noise and display a sample
        if self.offset == 0:
            generate_and_save_sample(imgs = generated_images, n = self.n, epoch = int(0 + self.offset), folder = self.folder, save = True)
        elif self.offset > 0:
            generate_and_save_sample(imgs = generated_images, n = self.n, epoch = int(0 + self.offset - 1), folder = self.folder, save = False)
        pass

    def on_epoch_end(self, epoch, logs = None):
        
        # run noise through current iteration of generator
        generated_images = self.model.generator(self.noise)
        generated_images = (generated_images * 127.5) + 127.5 # scale predicitons from [-1,1] to [0,255]
        generated_images.numpy()
        pass
        
        # generate images from random noise and display a sample
        if self.offset == 0:
            generate_and_save_sample(imgs = generated_images, n = self.n, epoch = int(epoch + 1 + self.offset), folder = self.folder, save = True)
        elif self.offset > 0:
            generate_and_save_sample(imgs = generated_images, n = self.n, epoch = int(epoch + self.offset), folder = self.folder, save = True)
        pass

def wgan_generate(model, n = 1, figsize = (4, 4), title_fsize = 10, show = True):
    """
    Function to generate images using a WGAN model (the generator component).
    
    :param tf.keras.Model model: WGAN model.
    :param int n: number of images generated. Only the first np.floor(np.sqrt(n))**2 images will be displayed.
    :param tuple figsize: Figure size dimensions.
    :param int title_fsize: Title font size.
    :param bool show: True if we want to display results.
    
    :return np.array gen_img: Generated images. Shape = (n, image_height, image_width, color_channels)
    """
    
    fig = plt.figure(figsize = figsize)
    
    # generate random noise as input and run it through the generator
    generated_images = model.generator(tf.random.normal([n, model.generator.input.shape[-1]]))
    generated_images = (generated_images * 127.5) + 127.5 # scale predicitons from [-1,1] to [0,255]
    gen_imgs = generated_images.numpy()

    # if we want to immediately display results
    if show:

        # display results
        g = int(np.floor(np.sqrt(n)))
        for i in range(int(g**2)):
            plt.subplot(g, g, i+1)
            plt.imshow(gen_imgs[i,]/255.)
            plt.axis('off')
            pass

        plt.suptitle('WGAN generated image(s)', fontsize = title_fsize)
        plt.show()
        pass
    
    return gen_imgs