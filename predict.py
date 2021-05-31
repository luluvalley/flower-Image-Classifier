# author: Xiaolu Rao
# Date: 2021-05-30
# Task: To uses a trained network to predict the class for an input image.

import PIL
from PIL import Image
import numpy as np
import json
import seaborn as sns

import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import matplotlib.pyplot as plt

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = "predict the network")
    
    parser.add_argument('--image_path',action="store", default = "flowers/test/100/image_07897.jpg", help='Load the test impage file for prediction.')
    parser.add_argument('--top_k',type=int, default = 3, help='Choose the top K matches to view.')
   # parser.add_argument('--cat_to_name', action="store", default='cat_to_name.json', help = "a mapping from category label to category name.")
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", help = "default setting with gpu")
 
    in_args = parser.parse_args()
    print("---- Test command Line Arguments... --- ", 
          "\n image_path = ", in_args.image_path,    
          "\n top_k = ", in_args.top_k,
         # "\n cat_to_name = ", in_args.cat_to_name,
          "\n gpu = ", in_args.gpu,
         )
    return in_args

# Loading the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epochs = checkpoint['epochs']
    batch_size = checkpoint['batch_size']
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])#, strict=False)
    optimizer = checkpoint['optimizer']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# Image Preprocessing
def process_image(image_path):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
    
    # Process a PIL image for use in a PyTorch model
    print("Process_image start...")
    
    prediction_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    img_pil = PIL.Image.open(image_path)
    img_tensor = prediction_transform(img_pil)
    print("done")
    return img_tensor.numpy()  

# Image show
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Class Prediction
def predict(image_path, model, topk=5):
#     ''' Predict the class (or classes) of an image using a trained deep learning model.
#     '''
    
    # Implement the code to predict the class from an image file
    
    # Pre-process the image
    img_np = process_image(image_path)
    
    # convert from numpy to tensor
    img_tensor = torch.from_numpy(img_np).type(torch.FloatTensor)
    
    # No need for GPU anymore
    model.cpu()
    img_tensor.cpu()
    
    img = img_tensor.unsqueeze_(0)
    
    # Start evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        logps = model.forward(img)
    
    # Covert to linear scale
    ps = torch.exp(logps)
    
    # Find the top 5 results
    top_p, top_class = ps.topk(topk, dim=1)
    
    # Detach all the details
    top_p = top_p.detach().numpy().tolist()[0]
    top_class = top_class.detach().numpy().tolist()[0]
    
    # Convert indices into classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Label mapping
    with open("cat_to_name.json", 'r') as f:
        cat_to_name = json.load(f)
    
    top_labels = [idx_to_class[lab] for lab in top_class]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_class]
    
    return top_p, top_labels, top_flowers
    
def main():
     # Get user input
    in_arg = get_input_args()

    # Loading the checkpoint
    model = load_checkpoint('checkpoint.pth')
    print("Loaded model = ", model)
    
    # Test image processing and image show 
    image_path = in_arg.image_path
    #image_test = process_image(image_path)
    #imshow(image_test, ax = None, title = None)
   
    # Predict
    top_p, top_labels, top_flowers = predict(image_path, model, topk=in_arg.top_k)
    
    # Print result:
    print("The top", in_arg.top_k, "flowers are:", top_flowers)
    print("And the probabilities are: ",top_p)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()   