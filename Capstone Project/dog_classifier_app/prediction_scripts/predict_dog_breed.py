# import packages
import numpy as np
import pickle
import re
import itertools
from os import listdir
from os.path import isfile, join
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K
from dog_classifier_app.prediction_scripts.detector_functions import face_detector, dog_detector
from dog_classifier_app.prediction_scripts.data_functions import path_to_tensor, extract_InceptionV3, inception_predict_breed


def make_prediction(img_path, checkpoint='static/weights.best.InceptionV3.hdf5', names_pkl='static/dog_names.pkl'):
    """
    Loads a trained network from a checkpoint file and determines whether the image contains a human, dog, or neither. 
    Then,
        - if a dog is detected in the image, returns the top predicted dog breed.
        - if a human is detected in the image, returns the top resembling dog breed.
        - if neither is detected in the image, provides output that indicates an error and asks user to use another image.
    
    Args:
        img_path - a file path to an image
        checkpoint - checkpoint filepath for trained CNN model (inception_model)
        names_pkl - path to pkl file that contains list of breeds
    Output: 
        - the input image
        - text showing whether the model detected a dog, human, or neither in the image (model is unsure)
        - text showing the top predicted dog breed and its probability
        
    """

    # clear Keras session to reset before loading model
    K.clear_session()

    # load InceptionV3 model for breed predictions
    model = load_model(checkpoint)

    # load list of dog breeds from pickle file
    with open(names_pkl, 'rb') as f:
        dog_names = pickle.load(f)

    # return dog breed that is predicted by the model
    prediction = inception_predict_breed(img_path, model, dog_names)

    # return prediction with message 
    if face_detector(img_path) == True and dog_detector(img_path) == False:
        # get list of filenames for breed images in breeds folder
        breed_files = [f for f in listdir('static/breeds') if isfile(join('static/breeds', f))]

        # get filename that matches breed prediction
        for filename in breed_files:
            if prediction.replace(' ', '_') in filename:
                breed_img_human = 'static/breeds/' + filename
        # return prediction message and brYoueed image
        return "Hey look! It's a ** HUMAN ** who looks a lot like this breed: {}. The resemblance is uncanny ;)".format(prediction), breed_img_human
    
    elif dog_detector(img_path) == True:
        # get list of filenames for breed images in breeds folder
        breed_files = [f for f in listdir('static/breeds') if isfile(join('static/breeds', f))]

        # get filename that matches breed prediction
        for filename in breed_files:
            if prediction.replace(' ', '_') in filename:
                breed_img_dog = 'static/breeds/' + filename
        return "Hey look! It's a ** DOG ** who looks a lot like this breed: {}.".format(prediction), breed_img_dog#'static/img/magic.jpg'
    
    else:
        return "Hmmm, not too sure about this one...¯\_(ツ)_/¯. Please try another image.", 'static/img/joke.jpg'