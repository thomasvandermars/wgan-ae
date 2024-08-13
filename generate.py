# IMPORT STATEMENTS
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

from wgan.model import *
from wgan.utils import *
from ae.model import *
from ae.utils import *

if __name__ == "__main__":
    
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description="generate_faces")
    parser.add_argument("-a", "--AE_name", type=str, default="AE", help="AE model name")
    parser.add_argument("-w", "--WGAN_name", type=str, default="WGAN", help="WGAN model name")
    parser.add_argument("-n", "--n", type=int, default=16, help="Number of images to generate")
    parser.add_argument("-t", "--title_fontsize", type=int, default=10, help="Title fontsize.")
    parser.add_argument("-f", "--fig_size", type=int, default=4, help="Figure size.")
    
    args = parser.parse_args()
    
    n = args.n
    figsize = (args.fig_size, args.fig_size)
    title_fsize = args.title_fontsize
    ae_name = args.AE_name
    wgan_name = args.WGAN_name
    
    # LOAD WGAN GENERATOR FOR GENERATING IMAGES
    generator = load_pretrained_model(file = wgan_name + "/generator.keras", folder = "trained_models")

    # LOAD ENCODER & DECODER FOR THE AE
    encoder = load_pretrained_model(file = ae_name + "/encoder.keras", folder = "trained_models")
    decoder = load_pretrained_model(file = ae_name + "/decoder.keras", folder = "trained_models")

    # CREATE AE
    ae = AE(encoder, decoder, 'temp_ae')

    # RUN RANDOM NOISE THROUGH WGAN-AE STACK
    generated_images = generator(tf.random.normal([n, generator.input.shape[-1]]))
    generated_images = ((generated_images * 127.5) + 127.5)/255. # scale predicitons from [-1,1] to [0,255]
    gen_imgs = generated_images.numpy()
    gen_imgs_deblur_denoise = ae(gen_imgs)

    # DISPLAY RESULTS
    fig = plt.figure(figsize = figsize)
    g = int(np.floor(np.sqrt(n)))
    for i in range(int(g**2)):
        plt.subplot(g, g, i+1)
        plt.imshow(gen_imgs_deblur_denoise[i,])
        plt.axis('off')
        pass

    plt.suptitle('WGAN-AE generated image(s)', fontsize = title_fsize)
    plt.savefig(os.getcwd() + '/generated_image(s).png')
    plt.show()