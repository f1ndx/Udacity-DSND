import numpy as np
import re
import random
import itertools
from sklearn.datasets import load_files  
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image                  
from tqdm import tqdm
from glob import glob


def path_to_tensor(img_path):
    """
    Takes a string-valued file path to a color image as input, preprocesses the data, and returns a 4D tensor suitable 
    for supplying to a Keras CNN.
    
    Args: 
        img_path - a string-valued file path to a color image
    Output: a 4D tensor
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_InceptionV3(tensor):
    """
    Extracts the bottleneck features corresponding to the pre-trained InceptionV3 model.
    """
    #return InceptionV3(weights='imagenet', include_top=False, pooling='avg').predict(preprocess_input(tensor))
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def inception_predict_breed(img_path, model, name_list):
    """
    Takes an image path, model, and breed list as input and returns the top dog breed predicted by the model.
    
    Args:
        img_path - a string-valued file path to an image
        model - a CNN model (e.g. inception_model)
        name_list - a full list of dog breeds
    Output:
        prediction - predicted dog breed with its probability
    """
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)[0]
    # return dog breed that is predicted by the model
    breed = name_list[np.argmax(predicted_vector)].replace('_',' ')
    prob = predicted_vector[np.argmax(predicted_vector)]
    #prediction = f"{breed} (probability = {prob:.2})"
    return breed
