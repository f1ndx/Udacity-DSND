# Author: Anastasia Atanasoff
# train.py does the following:
#       - successfully trains a new network on a dataset of images and saves the model to a checkpoint
#       - allows users to choose vgg16 (default) or densenet121 architectures available from torchvision.models
#       - allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
#       - allows users to choose training the model on a GPU
# Basic usage:  python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
#       - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#       - Choose architecture: python train.py data_dir --arch 'densenet121'
#       - Set hyperparameters: python train.py data_dir --lr 0.01 --hidden_units 512 --epochs 20
#       - Use GPU for training: python train.py data_dir --gpu

# Imports
import argparse
import numpy as np
import torch
import torchvision
import os
import time
import json
import copy

from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler

use_gpu = torch.cuda.is_available() # CUDA availability

# main() function
def main():
    args = get_arguments()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = os.path.join(args.save_dir, 'checkpoint.pth')

    # Load a pre-trained network
    if args.arch == 'vgg16': 
        num_features = model.classifier[0].in_features
        model = models.vgg16(pretrained = True)
    elif args.arch == 'densenet121':
        num_features = 1024
        model = models.densenet121(pretrained = True)
    else:
        print('Unexpected network architecture')
        sys.exit()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define new untrained feed-forward network as a classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, args.hidden_units)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p = 0.5)),
                              ('hidden', nn.Linear(args.hidden_units, args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim = 1)),
                              ]))
    
    model.classifier = classifier
    data_transforms, image_datasets, dataloaders = load_data(args.data_dir)
    
    if args.cuda:
        # If CUDA is available
        if use_gpu:
            model = model.cuda()
            print('Using GPU: '+ str(use_gpu))
        else:
            print('Using CPU since GPU is not available/configured')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr = args.lr)  
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

    # Train model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, epochs = args.epochs)

    # Validate model
    validate_model(model)

    checkpoint = {'batch_size': 8,
                  'lr': args.lr,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'epochs': args.epochs,
                  'hidden_units': args.hidden_units,
                  'class_to_idx': image_datasets['train'].class_to_idx
                }

    # Save checkpoint
    torch.save(checkpoint, 'checkpoint.pth')
    
# get_arguments() function
def get_arguments():
    parser = argparse.ArgumentParser(description='Flower Classification Trainer')
    
    parser.add_argument('--save_dir', type = str, action = 'store', dest = 'save_dir', default = '.', help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, action = 'store', dest = 'arch', default = 'vgg16', help = "Set architechture ('vgg16' or 'densenet121')")
    parser.add_argument('--lr', type = float, action = 'store', dest = 'lr', default = 0.01, help = 'Set learning rate')
    parser.add_argument('--hidden_units', type = int, action = 'store', dest = 'hidden_units', default = 512, help = 'Set number of hidden units')
    parser.add_argument('--epochs', type = int, action = 'store', dest = 'epochs', default = 8, help = 'Set number of epochs')
    parser.add_argument('--gpu', action = 'store_true', dest = 'cuda', default = False, help = 'Use CUDA for training')
    parser.add_argument('data_dir', type = str, action = 'store', default='flowers', help='dataset directory')
    
    return parser.parse_args()

# load_data() function
def load_data(data_dir):
    args = get_arguments()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    batch_size = 8
    #use_gpu = torch.cuda.is_available()
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = batch_size, shuffle = True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = batch_size),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = batch_size)
    }
    
    return data_transforms, image_datasets, dataloaders

# train_model() function
def train_model(model, criterion, optimizer, scheduler, epochs): 
    args = get_arguments()
    data_transforms, image_datasets, dataloaders = load_data(args.data_dir)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = data
                
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
# validate_model() function
def validate_model(model):
    args = get_arguments()
    data_transforms, image_datasets, dataloaders = load_data(args.data_dir)
    criterion = nn.CrossEntropyLoss()
  
    model.eval()
    accuracy = 0
    testing_loss = 0

    for data in dataloaders['test']:
        inputs, labels = data

        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        
        output = model.forward(inputs)
        testing_loss += criterion(output, labels).item()
        
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    print('Validation loss: {:.3f}'.format(testing_loss / len(dataloaders['test'])),
          'Accuracy: {:.3f}'.format(accuracy / len(dataloaders['test'])))
    
# Run main()
if __name__ == '__main__':
    main()