# author: Xiaolu Rao
# Date: 2021-05-30
# Task: To train a new network on a dataset and save the model as a checkpoint.

# Imports python modules
from time import time, sleep
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# import PIL
# from PIL import Image
# import numpy as np
# import json
# import seaborn as sns


def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = "train the network")
    
    parser.add_argument('--train_dir',  default = 'flowers/train', help = 'path to the folder of train data')
    parser.add_argument('--valid_dir',  default = 'flowers/valid', help = 'path to the folder of valid data')
    parser.add_argument('--test_dir',  default = 'flowers/test', help = 'path to the folder of test data')
    parser.add_argument('--arch', default = 'vgg16', help = 'path to the training model type and the default is VGG16' )
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", help="set the directory to save checkpoints")
    parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.001, help = "default learning rate")
    parser.add_argument('--h1', type=int, dest="h1", action="store", default=900, help = "Hidden layer 1")
    parser.add_argument('--h2', type=int, dest="h2", action="store", default=300, help = "Hidden layer 2")
    parser.add_argument('--epochs', type=int, dest="epochs", action="store",  default=3, help = "default epoch loop")
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", help = "default setting with gpu")
 
    in_args = parser.parse_args()
    print("---- Test command Line Arguments... --- \n train_dir = ", in_args.train_dir, 
          "\n valid_dir = ", in_args.valid_dir,
          "\n test_dir = ", in_args.test_dir,    
          "\n save_dir = ", in_args.save_dir,
          "\n arch = ", in_args.arch,
          "\n h1 = ", in_args.h1,
          "\n h2 = ", in_args.h2,
          "\n epochs = ", in_args.epochs,
          "\n gpu = ", in_args.gpu,
         )
    return in_args

# For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. 
# This will help the network generalize leading to better performance. 
# You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
def train_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    return train_transforms

# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. 
# For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
def test_transforms():
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    return test_transforms

# Load the datasets with ImageFolder
def load_datasets(dir, transforms):
    data = datasets.ImageFolder(dir, transform=transforms)
    #print("\n---- Test datasets", dir ," datasets... \n", data)
    return data
   
# Using the image datasets and the trainforms, define the dataloaders
def load_data(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader

#Load a pre-trained network: use the pretrained network VGG16.
def load_model(model_name):
    resnet18 = models.resnet18(pretrained=True)
    vgg13 = models.vgg13(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    load_models = {'resnet18': resnet18, 'vgg13': vgg13, 'vgg16': vgg16}

    model = load_models[model_name]
       
    print("\n---- Test loaded pre-trained model:",model)
    return model

# Define a new network as a new classifier,
def update_model(model, learning_rate, h1, h2):
    #-- IMPORTANT: Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device = ", device)
    
    # Define a new network as a new classifier, 
    from collections import OrderedDict
    new_classifier = nn.Sequential(OrderedDict([
                                    ('input', nn.Linear(25088, h1)),
                                    ('relu1', nn.ReLU()),
                                    ('drop1', nn.Dropout(0.2)),
                                    ('h1',nn.Linear(h1, h2)),
                                    ('relu2', nn.ReLU()),
                                    ('h2',nn.Linear(h2, 102)), ## IMPORTANT: The output is 102 flower categories.
                                    ('output', nn.LogSoftmax(dim=1))                           
                                    ]))

    model.classifier = new_classifier
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device);
    print("Current model is: \n", model)
    
    return model, optimizer, criterion, device
    
def train_network(Model, Epochs, Trainloader, Validloader, Optimizer, Criterion, Device):
    print("\n---- Training start ...")
    model = Model
    epochs = Epochs
    trainloader = Trainloader
    validloader = Validloader
    optimizer = Optimizer
    criterion = Criterion
    device = Device
        
    steps = 0
    running_loss = 0
    print_every = 5
    
    #train_losses, test_losses = [], [] # these two list are used for drawing later.

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
           
            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)

            # Compute loss
            loss = criterion(logps, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameter
            optimizer.step()

            # Calculate the total loss
            running_loss += loss.item()

            # Compute the accuracy
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0

                # put model in evalaution mode
                model.eval()

                # turn off gradients for validation to save memory and computations.
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy based on validation data
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                #train_losses.append(running_loss/len(trainloader))        
               # test_losses.append(test_loss/len(validloader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0

                # put model back in training mode
                model.train()


    print("-- Training is done --") 
    return model

# Do validation on the test set
def test_network(model, testloader, device):
    print("calculate the accuracy based on the test data...")

    correct = 0
    total = 0

    # put model in evalaution mode
    model.eval()

    # turn off gradients for validation to save memory and computations.
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("-- test done -- ")
    print("The accuracy on the test image: %d %%" % (100 * correct / total))
    
def save_checkpoint(trained_model, train_data, epochs, optimizer, model_name):
    model = trained_model
    model.class_to_idx = train_data.class_to_idx
    
    torch.save({
            'input_size':25088,
            'output_size': 102,
            'epochs': epochs,
            'batch_size': 32,
            'model': load_model(model_name),
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }, 
            'checkpoint.pth')
    

# Main program function defined below
def main():
    start_time = time()
    print("start_time is: ",start_time)
    
    # Get user input
    in_arg = get_input_args()
    
    # Set dir for training
    train_dir = in_arg.train_dir
    valid_dir = in_arg.valid_dir
    test_dir = in_arg.test_dir
    
    # Load the datasets 
    train_data = load_datasets(train_dir, train_transforms())
    test_data = load_datasets(test_dir, test_transforms())
    valid_data = load_datasets(valid_dir, test_transforms())
    
    # Define the dataloaders
    trainloader = load_data(train_data)
    testloader = load_data(test_data, train = False)
    validloader = load_data(valid_data, train = False)
    
    # Define the model
    model = load_model(in_arg.arch)
    
    # Update the classifier
    model, optimizer, criterion, device = update_model(model, in_arg.learning_rate, in_arg.h1, in_arg.h2)
    
    # Train the network based on updated model and dataset
    from workspace_utils import active_session
    with active_session():
        trained_model = train_network(model, in_arg.epochs, trainloader, validloader, optimizer, criterion, device)
    
    # Test the network
    test_network(trained_model, testloader, device)

    # Save the checkpoint
    save_checkpoint(trained_model, train_data, in_arg.epochs, optimizer, in_arg.arch)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()