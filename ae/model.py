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

def initialize_encoder(input_dims = (64, 64, 3), 
                       max_filters = 1024, 
                       kernel_size = 5, 
                       latent_dim = 128,
                       leaky_relu_slope = 0.2,
                       model_name = 'encoder'):
    """
    Function to initialize the encoder architecture according to specified parameters.
    
    :param tuple input_dims: Input dimensions for images (img_height, img_width, color_channels).
    :param int max_filters: Maximum number of filters for the top layer in the encoder.
    :param int kernel_size: Kernel size used.
    :param int latent_dim: Number of elements in the latent space.
    :param float leaky_relu_slope: negative slope of leaky ReLu activation. If None, the model uses Parametric ReLU (PReLU) 
                                   and attempts to arrive at the slope through learnable parameters. Defaults to 0.2.
    :param str model_name: Name of model.
    
    :return: tf.keras.Sequential discriminator: encoder architecture.
    """

    # do a little math to determine the number of upsamplings required in both the height and width
    # dimensions to arrive from the input dimensions to the desired feature map
    n_upsamples_height = (tf.math.log(float(input_dims[0]))/tf.math.log(2.))-2
    n_upsamples_width = (tf.math.log(float(input_dims[1]))/tf.math.log(2.))-2
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
        elif leaky_relu_slope == 0: 
            layer_list.append(ReLU(name = "ReLu_"+str(i+1)))
        else:
            layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu_"+str(i+1)))
        f = int(f*2)
        pass

    layer_list.append(Flatten())
    layer_list.append(Dense(latent_dim, name = "FC_out"))
    layer_list.append(BatchNormalization(name = "BN_final_out"))
    if leaky_relu_slope == None:
        layer_list.append(PReLU(name = "PReLu_out"))
    elif leaky_relu_slope == 0: 
        layer_list.append(ReLU(name = "ReLu_out"))
    else:
        layer_list.append(LeakyReLU(leaky_relu_slope, name = "lky_ReLu_out"))

    encoder = Sequential(layer_list, name = model_name)

    return encoder

def initialize_decoder(input_dims = (64, 64, 3), 
                       max_filters = 1024, 
                       kernel_size = 5, 
                       latent_dim = 128,
                       leaky_relu_slope = None,
                       model_name = 'decoder'):
    """
    Function to initialize the decoder architecture according to specified parameters.
    
    :param tuple input_dims: Input dimensions for images (img_height, img_width, color_channels).
    :param int max_filters: Maximum number of filters for the lowest layer in the decoder.
    :param int kernel_size: Kernel size used.
    :param int latent_dim: Number of elements in the latent space.
    :param float leaky_relu_slope: negative slope of leaky ReLu activation. If None, the model uses Parametric ReLU (PReLU) 
                                   and attempts to arrive at the slope through learnable parameters. Defaults to None.
    :param str model_name: Name of model.
    
    :return: tf.keras.Sequential generator: decoder architecture.
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
    layer_list.append(Input(shape=(latent_dim,), name = "Input"))

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
                             activation = "sigmoid", kernel_initializer = "glorot_normal", name = "TConv2D_out"))

    decoder = Sequential(layer_list, name = model_name)

    return decoder
    
def AE_loss(y_true, y_pred):
    loss = y_true * tf.math.log(1e-10 + y_pred) + (1.0 - y_true) * tf.math.log(1e-10 + 1.0 - y_pred)
    loss = tf.reduce_mean(tf.multiply(-1.0, tf.reduce_sum(loss, axis = [1,2,3])))
    return loss



class AE(Model):
    def __init__(self, encoder, decoder, model_name = "AE"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name = "loss")
        self._name = model_name
        pass

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        
        noisy_imgs, target_imgs = data
        
        with tf.GradientTape() as tape:
            
            z = self.encoder(noisy_imgs) # run noisy images through encoder
            reconstructions = self.decoder(z) # run latent space through decoder to get reconstructed images
            
            # calculate reconstruction autoencoder loss
            #loss = AE_loss(y_true = target_imgs, y_pred = reconstructions)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target_imgs, reconstructions)), axis = [1,2,3]))
            pass
        
        # get the gradients and backpropogate the loss
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    def call(self, inputs, training = False):
        
        z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        return reconstruction

def show_training_sample(noisy_imgs, reconstructions, target_imgs, epoch, folder, save = True):
    
    # clear existing console output
    display.clear_output(wait = True)

    # construct image array
    fig = plt.figure(figsize = (7, 9))

    # display results
    l = 0
    p = np.array([1,4,7,10])
    for i in p:
        plt.subplot(4, 3, i)
        if i == 1:
            plt.title('Blurry/Noisy Inputs', fontsize = 8)
        plt.imshow(noisy_imgs[l,])
        l = l + 1
        plt.axis('off')
        pass

    l = 0
    for i in (p+1):
        plt.subplot(4, 3, i)
        if i == 2:
            plt.title('Reconstructions AE', fontsize = 8)
        plt.imshow(reconstructions[l,])
        l = l + 1
        plt.axis('off')
        pass

    l = 0
    for i in (p+2):
        plt.subplot(4, 3, i)
        if i == 3:
            plt.title('Target Images', fontsize = 8)
        plt.imshow(target_imgs[l,])
        l = l + 1
        plt.axis('off')
        pass

    plt.suptitle('De-Blur / De-Noise AE - epoch #' + str(int(epoch)), fontsize = 12)
    if save:
        plt.savefig(os.getcwd() + '/' + folder + '/images_at_epoch_{:04d}.png'.format(epoch))
        pass
    plt.show()
    pass

class monitor_ae_performance(Callback):
    
    def __init__(self, data_gen, folder = 'training_progress'):
        self.data_gen = data_gen
        self.batch_size = self.data_gen.batch_size
        self.length_data = int(np.floor(len(self.data_gen.filenames)/self.batch_size))
        self.folder = folder
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
        
        # run a batch through the autoencoder model to get reconstructions
        noisy_imgs, target_imgs = self.data_gen.__getitem__(0)
        z = self.model.encoder(noisy_imgs)
        reconstructions = self.model.decoder(z)
        
        # construct, show and save a sample of side-by-side noisy input, ae reconstructions, and the target images
        if self.offset == 0:
            show_training_sample(noisy_imgs, reconstructions, target_imgs, int(0 + self.offset), self.folder, True)
        elif self.offset > 0:
            show_training_sample(noisy_imgs, reconstructions, target_imgs, int(0 + self.offset - 1), self.folder, False)
        pass
    
    def on_epoch_end(self, epoch, logs = None):
        
        # run a batch through the autoencoder model to get reconstructions
        noisy_imgs, target_imgs = self.data_gen.__getitem__(0)
        z = self.model.encoder(noisy_imgs)
        reconstructions = self.model.decoder(z)
        
        # construct, show and save a sample of side-by-side noisy input, ae reconstructions, and the target images
        if self.offset == 0:
            show_training_sample(noisy_imgs, reconstructions, target_imgs, int(epoch + 1 + self.offset), self.folder, True)
        elif self.offset > 0:
            show_training_sample(noisy_imgs, reconstructions, target_imgs, int(epoch + self.offset), self.folder, True)
        
        #### calculate test loss ####
        
        # run a randomly selected test batch through the autoencoder
        noisy_imgs, target_imgs = self.data_gen.__getitem__(np.random.choice(range(self.length_data), 1)[0])
        z = self.model.encoder(noisy_imgs)
        reconstructions = self.model.decoder(z)
        
        # calculate reconstruction autoencoder loss
        #loss = AE_loss(y_true = target_imgs, y_pred = reconstructions)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target_imgs, reconstructions)), axis = [1,2,3]))    
        
        print("Test Reconstruction Loss: " + str(loss))
        
        pass