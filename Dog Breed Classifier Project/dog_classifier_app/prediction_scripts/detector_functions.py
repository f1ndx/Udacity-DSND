import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from dog_classifier_app.prediction_scripts.data_functions import path_to_tensor

def face_detector(img_path):
    """
    Detects whether or not an image is predicted to contain a human face.
    
    Args:
        img_path - a string-valued file path to an image
    Output: boolean value - returns True if a human face is detected in an image and False otherwise
    """

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_alt.xml')

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


def dog_detector(img_path):
    """
    Detects whether or not an image is predicted to contain a dog by the pre-trained ResNet-50 model.
    
    Args: 
        img_path - a string-valued file path to an image
    Output: boolean value - returns True if a dog is detected in an image (and False if not)
    """

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # preprocesses an image to supply to ResNet-50
    img = preprocess_input(path_to_tensor(img_path))

    # returns a prediction vector for image located at img_path
    prediction = np.argmax(ResNet50_model.predict(img))

    return ((prediction <= 268) & (prediction >= 151))