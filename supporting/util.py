"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2imVisualizer(input_image, imtype=np.int32):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = (image_numpy-np.amin(image_numpy))/(np.amax(image_numpy)-np.amin(image_numpy))
        
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1)) 
        
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
        
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


#def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    
#    image_pil = Image.fromarray(image_numpy,'I')
#    image_pil.save(image_path)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def GetImageCentroid(Image):
    """
    Get Centroid of a binary image
    Parameters:
        Image: a binary image
    """
    ImArr = np.array(Image)
    NZ = np.nonzero(ImArr)
    MedY = np.median(NZ[0])
    MedX = np.median(NZ[1])
    Centroid = (MedX,MedY)
    return Centroid
    
def FitResizedImage(A,B,ResizeVal,w2,h,load_size):
    """
    Resizes the images and ensures the resized images are the same dimensions as the original images
    
    Parameters
    ----------
    A : PIL Image
        Input Image.
    B : PIL Image
        Ground Truth Image.
    ResizeVal : float
        Value images will be rescaled by.
    w2 : int
        width of Images A and B.
    h : int
        height of Images A and B.
    load : int
        width and height of the final image to be cropped
    Returns
    -------
    ACropped : PIL Image
        Resized and Cropped images with same dimensions as input image A.
    BCropped : PIL Image
        Resized and Cropped images with same dimensions as input image B.

    """
    SizeNew = tuple([int(ResizeVal*x) for x in A.size])
    
    AResized = A.resize((SizeNew[0],SizeNew[1]))
    BResized = B.resize((SizeNew[0],SizeNew[1]))
    
    GTCentroid = GetImageCentroid(BResized)
        
    if np.isnan(GTCentroid[0]):
        GTCentroid = (load_size/2,load_size/2)
        
    StartX = int(GTCentroid[0] - load_size/2)
    EndX = int(GTCentroid[0] + load_size/2)
    StartY = int(GTCentroid[1] - load_size/2)
    EndY = int(GTCentroid[1] + load_size/2)
    
    if EndX-StartX != load_size:
        EndX = StartX + load_size
    if EndY-StartY != load_size:
        EndX = StartY + load_size
      
    if StartX < 0:
        EndX = EndX - StartX
        StartX = 0 
    if StartY < 0:
        EndY = EndY - StartY
        StartY = 0         
    if EndX > SizeNew[0]:
        StartX = StartX - (EndX-SizeNew[0])
        EndX = SizeNew[0]    
    if EndY > SizeNew[1]:
        StartY = StartY - (EndY-SizeNew[1])
        EndY = SizeNew[1]           
    
    ACropped = AResized.crop((StartX,StartY,EndX,EndY))
    BCropped = BResized.crop((StartX,StartY,EndX,EndY))
    try:
    
        assert ACropped.size[0] == load_size and ACropped.size[1] == load_size
        assert BCropped.size[0] == load_size and BCropped.size[1] == load_size
    except:
        print(GTCentroid)
        print(ACropped.size[0])
        print(ACropped.size[1])
        print(StartX)
        print(EndX)
        print(StartY)
        print(EndY)
        assert ACropped.size[0] == load_size and ACropped.size[1] == load_size
        assert BCropped.size[0] == load_size and BCropped.size[1] == load_size
    return ACropped,BCropped
    
    
    