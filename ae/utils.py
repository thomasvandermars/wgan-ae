import imageio
import os
import tensorflow as tf
import cv2
from tqdm import tqdm

from sklearn.model_selection import train_test_split

def split_image_data(test_size = 0.05, folder = 'data'):
    """
    Function to split the image data set into train and test sets.
    
    :param float test_size: percentage of total dataset assigned to test set. Defaults to 0.05.
    :param str folder: name of subdirectory within the project directory that holds the data. Default to "data"
    
    :return list train_set: training data (list of full pathnames to images)
    :return list test_set: test data (list of full pathnames to images)
    """
    # extract list of filenames for images in dataset
    filenames = [os.getcwd() + '/' + folder + '/' + x for x in os.listdir(os.getcwd() + '/' + folder)]
    
    # split into train and test set
    train_set, test_set, _, _ = train_test_split(filenames, filenames, test_size = 0.15, random_state=42)
    
    print('overall dataset size: ', str(len(filenames)))
    print('-------------------------------')
    print('train dataset size: ', str(len(train_set)))
    print('test dataset size: ', str(len(test_set)))
    
    return train_set, test_set

def load_pretrained_model(file, folder = "trained_models"):
    """
    Function to load a pretrained model from file.
    
    :param str file: filename (.keras).
    :param str folder: subdirectory within the project directory that holds the file. Default to "trained_models"
    
    :return: tf.keras.Sequential model
    """
    
    if not os.path.exists(os.getcwd() + "/" + folder + "/" + file):
        print("File was not found! Please ensure the correct file and folder name...")
    else:
        model = tf.keras.models.load_model(os.getcwd() + "/" + folder + "/" + file, compile = False)
        print("Model: [" + str(model.name) + "] was successfully loaded!")
        return model

def save_ae_model(model, folder = "trained_models"):
    """
    Function to save a trained model to file.
    
    :param tf.keras.Model model: WGAN model
    :param str file: filename (.keras).
    :param str folder: subdirectory within the project directory that holds the file. Default to "trained_models"
    
    :return: None
    """
    
    # makes sure that we avoid the "not compiled warnings"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    model.encoder.save(os.getcwd() + "/" + folder + "/" + model.name + "/encoder.keras")
    model.decoder.save(os.getcwd() + "/" + folder + "/" + model.name + "/decoder.keras")
    pass

def create_gif(fname, train_progress_folder, loop = 30, duration = 100):
    """
    Function to save training progress images as a gif file.
    
    :param str fname: filename
    :param str train_progress_folder: subdirectory within the project directory that holds images displaying training progress.
    :param int loop: number of loops
    :param float duration: duration per frame
    
    :return: None
    """
    
    # extract filename list
    files = [os.getcwd() + '/' + train_progress_folder + '/' + x for x in os.listdir(os.getcwd() + '/' + train_progress_folder)]
    files = sorted(files)

    file_list = []
    for f in tqdm(files):
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        file_list.append(image)
        pass
    
    # creat gif
    imageio.mimsave(os.getcwd() + '/' + fname, file_list, format = 'GIF', loop = loop, duration = duration)
    pass