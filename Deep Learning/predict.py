# Author: Anastasia Atanasoff
# predict.py does the folllowing:
#       - successfully reads in an image and a checkpoint, then prints the most likely image class and it's associated probability.
#       - allows users to print out the top K (5 is default) classes along with associated probabilities.
#       - allows users to load a JSON file that maps the class values to other category names.
#       - allows users to use the GPU to calculate the predictions.
# Basic usage:  python predict.py /path/to/image checkpoint
# Options:
#       - Return top K most likely classes: python predict.py image_path checkpoint.pth --topk 3
#       - Use a mapping of categories to real names: python predict.py image_path checkpoint.pth --category_names cat_to_name.json
#       - Use GPU for inference: python predict.py image_path checkpoint.pth --gpu

import argparse
import json
import numpy as np
import torch
import os

from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms, utils

use_gpu = torch.cuda.is_available() # CUDA availability

# main() function
def main():
    args = get_arguments()
                
    model = load_checkpoint(args.checkpoint)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
      
    probs, classes = predict(args.image_path, model, topk=int(args.topk))
    print('\nPREDICTION: Most likely image class is "{}" and it\'s associated probability is {}.'.format([cat_to_name[x] for x in classes][0], round(probs[0], 3)))
    print('\n* Top {} predicted probabilities: {}'.format(int(args.topk), probs))
    print('* Top {} predicted classes: {}'.format(int(args.topk), classes))
    print('* Associated class names: ', [cat_to_name[x] for x in classes])
 
    
# get_arguments() function    
def get_arguments():
    parser = argparse.ArgumentParser(description = 'Flower Classification Predictor')
    
    # Required
    parser.add_argument('image_path', type = str, action = 'store', help = 'Path of image')
    parser.add_argument('checkpoint' , type = str, action = 'store', default = 'checkpoint.pth', help = 'Path of checkpoint model')
    
    parser.add_argument('--gpu', action = 'store_true', dest = 'cuda', default = False, help = 'Use CUDA for training')
    parser.add_argument('--category_names' , type = str, action = "store", dest = "category_names", default = 'cat_to_name.json', help = "Path of mapper that maps category label to category name")
    parser.add_argument("--topk", type = int, action = "store", dest = "topk", default = 5, help = "Display top k probabilities")
    return parser.parse_args()

# load_checkpoint() function
def load_checkpoint(filename):
    ''' Loads a checkpoint and rebuilds the model.
    '''
    if os.path.isfile(filename):
        print(f'Loading checkpoint {filename}')
        if use_gpu:
            checkpoint = torch.load(filename)
        else:
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        
        if checkpoint['arch'] == 'vgg16':
            model = models.vgg16(pretrained = True)        
        elif checkpoint['arch'] == 'densenet121':
            model = models.densenet121(pretrained = True)
            
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']

        return model
    else:
        print(f'No checkpoint found at: {filename}')
   
# process_image() function
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''   
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    image_size = pil_image.size
    shortest_side = min(image_size)
    pil_image = pil_image.resize((int(((image_size[0]/shortest_side) * 256)), int((image_size[1]/shortest_side) * 256)))
    value = 0.5 * (256 - 224)
    pil_image = pil_image.crop((value, value, 256-value, 256-value))
    np_image = np.array(pil_image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

# predict() function
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    if use_gpu:
        model.cuda()
        print('Using GPU: '+ str(use_gpu))
    else:
        print('Using CPU since GPU is not available/configured')
        
    model.eval()

    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    image = Variable(image)
      
    if use_gpu:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk probabilites and indexes
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])

    return prob, label

if __name__ == "__main__":
    main()