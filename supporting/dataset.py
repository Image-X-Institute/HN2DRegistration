# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:12:02 2025

@author: mgar5380
"""
import torch
import random
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from supporting.image_folder import make_dataset,make_DVF_dataset

from skimage.registration import optical_flow_tvl1

class CustomImageDVFDataset(Dataset):
    def __init__(self,fixedDir,movingDir,DVFDir='',imageSize=512,preCropImageSize=512,
                 MaxRotate=10.0,FlipFlag=False,phase='train',datasetSize=float('inf')):
        self.fixed_paths = sorted(make_dataset(fixedDir))
        self.moving_paths = sorted(make_dataset(movingDir))
        self.phase = phase
        if self.phase == 'train':
            self.DVF_paths = sorted(make_DVF_dataset(DVFDir))
        else:
            self.DVF_paths = None
        self.imageSize = imageSize
        self.preCropImageSize = preCropImageSize
        self.MaxRotate = MaxRotate
        self.FlipFlag = FlipFlag
        self.datasetSize = datasetSize
        
        
    def __len__(self):
        """Return the total number of images."""
        return min(len(self.fixed_paths),self.datasetSize)
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.        
        """
        
        fixed_path = self.fixed_paths[index]
        moving_path = self.moving_paths[index]    
        if self.phase == 'train':
            dvf_path = self.DVF_paths[index]
        #print(fixed_path)
        #print(moving_path)
        #print(dvf_path)
        fixedImage = Image.open(fixed_path)
        movingImage = Image.open(moving_path)
        if self.phase == 'train':
            if '.mha' in dvf_path:
                DVFImage = sitk.ReadImage(dvf_path)
                DVFSpacing = np.array(DVFImage.GetSpacing())
                DVFLoad = sitk.GetArrayFromImage(DVFImage)
                DVFNew = np.transpose(DVFLoad[:,:,:,[0,1]],[2,1,3,0])
                idx = [1,0,2]
                DVFSpacing = DVFSpacing[idx]
                shape = DVFNew.shape
                DVF = np.reshape(DVFNew,(shape[0],shape[1],shape[2]))
            else:
                DVF = np.load(dvf_path)
                DVFSpacing = np.array([0.388,0.388])
        
        fixedImArr = np.asarray(fixedImage)
        movingImArr = np.asarray(movingImage)
        
        fixedImage = Image.fromarray(fixedImArr.astype(dtype=np.float32))
        movingImage = Image.fromarray(movingImArr.astype(dtype=np.float32))
        if self.phase == 'train':
            #DVFUImage = Image.fromarray(DVF[:,:,0]*(1/DVFSpacing[1]).astype(dtype=np.float32))  
            #DVFVImage = Image.fromarray(-DVF[:,:,1]*(1/DVFSpacing[0]).astype(dtype=np.float32)) 
            DVFUImage = Image.fromarray(-DVF[:,:,0]*(1/DVFSpacing[0]))
            DVFVImage = Image.fromarray(DVF[:,:,1]*(1/DVFSpacing[1]))
        
        w,h = fixedImage.size
        
        #Do Additional Data Augmentation
        if abs(self.MaxRotate) > 0:
            RotVal = random.uniform(-self.MaxRotate,self.MaxRotate)  
        else:
            RotVal = 0
        
        if abs(self.MaxRotate) > 0:
            #fixedImage = fixedImage.rotate(RotVal,resample=Image.Resampling.BILINEAR)
            #movingImage = movingImage.rotate(RotVal+RotShift,resample=Image.Resampling.BILINEAR)
            fixedImage = fixedImage.rotate(RotVal,resample=Image.Resampling.BICUBIC)
            movingImage = movingImage.rotate(RotVal,resample=Image.Resampling.BICUBIC)
            if self.phase == 'train':
                DVFUImage = DVFUImage.rotate(RotVal,resample=Image.Resampling.BICUBIC)
                DVFVImage = DVFVImage.rotate(RotVal,resample=Image.Resampling.BICUBIC)
        
        #Random Cropping
        if (w > self.preCropImageSize) or (h > self.preCropImageSize):
            startX = random.randrange(0, (w - self.preCropImageSize-1), 1)  
            startY = random.randrange(0, (h - self.preCropImageSize-1), 1) 
        
            endX = startX + self.preCropImageSize
            if endX >= w:
                endX = w-1
                startX = endX-self.preCropImageSize
            endY = startY + self.preCropImageSize
            if endY >= h:
                endY = h-1
                startY = endY-self.preCropImageSize
            
            fixedImage = fixedImage.crop((startX, startY, startX + self.preCropImageSize, startY + self.preCropImageSize))
            movingImage = movingImage.crop((startX, startY, startX + self.preCropImageSize, startY + self.preCropImageSize))
            if self.phase == 'train':
                DVFUImage = DVFUImage.crop((startX, startY, startX + self.preCropImageSize, startY + self.preCropImageSize))
                DVFVImage = DVFVImage.crop((startX, startY, startX + self.preCropImageSize, startY + self.preCropImageSize))

        if self.FlipFlag:       
            flip = random.random() > 0.5
            
            if flip:
                fixedImage = fixedImage.transpose(Image.FLIP_TOP_BOTTOM)
                movingImage = movingImage.transpose(Image.FLIP_TOP_BOTTOM)        
                if self.phase == 'train':
                    DVFUImage = DVFUImage.transpose(Image.FLIP_TOP_BOTTOM)
                    DVFVImage = DVFVImage.transpose(Image.FLIP_TOP_BOTTOM)  

        fixedImage = fixedImage.resize((self.imageSize,self.imageSize))
        movingImage = movingImage.resize((self.imageSize,self.imageSize))
        if self.phase == 'train':
            DVFUImage = DVFUImage.resize((self.imageSize,self.imageSize))
            DVFVImage = DVFVImage.resize((self.imageSize,self.imageSize))
            DVFUArr = np.array(DVFUImage)
            #print(shape[0]/imageSize)
            ScalingFactor = (self.preCropImageSize/self.imageSize)
            #print(ScalingFactor)
            DVFUImage = Image.fromarray(DVFUArr/(ScalingFactor))
            DVFVArr = np.array(DVFVImage)
            #print(shape[1]/imageSize)
            DVFVImage = Image.fromarray(DVFVArr/(ScalingFactor))
        
        fixedArray = np.asarray(fixedImage)
        fixedArray = (fixedArray - np.min(fixedArray)) / (np.max(fixedArray) - np.min(fixedArray))
        movingArray = np.asarray(movingImage)
        movingArray = (movingArray - np.min(movingArray)) / (np.max(movingArray) - np.min(movingArray))
        
        fixed_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
        fixed_array[0, :, :] = np.asarray(fixedArray)
        moving_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
        moving_array[0, :, :] = np.asarray(movingArray)

        target_flowArr = np.zeros((2, self.imageSize, self.imageSize), dtype=np.float32)
        if self.phase == 'train':
            target_flowArr[0,:,:] = np.asarray(DVFUImage)
            target_flowArr[1,:,:] = np.asarray(DVFVImage)
    
        fixed_tensor = torch.from_numpy(fixed_array)
        moving_tensor = torch.from_numpy(moving_array)
        
        target_flow = torch.from_numpy(target_flowArr)

        return {'fixed': fixed_tensor, 'moving': moving_tensor, 'target_flow':target_flow,'A_paths': fixed_path, 'B_paths': moving_path}

class CustomImageDataset(Dataset):
    def __init__(self,fixedDir,movingDir='',imageSize=512,MaxShift=60,
                 MaxRotate=10.0,RotShift=3.0,FlipFlag=True,PerspectiveScale=0.5,
                 CalculateDVF=False,preCropImageSize=512,maskDir='',phase='train',OARDir=''):
        self.phase = phase
        self.fixed_paths = sorted(make_dataset(fixedDir))
        if not movingDir:
            self.moving_paths = self.fixed_paths
        else:
            self.moving_paths = sorted(make_dataset(movingDir))
        self.MaskFlag = False
        if not maskDir:
            self.mask_paths=[]
        else:
            self.mask_paths = sorted(make_dataset(maskDir))
            if self.phase == 'test':
                self.MaskFlag = True
        if not OARDir:
            self.OAR_paths=[]
        else:
            self.OAR_paths = sorted(make_dataset(OARDir))
            if self.phase == 'test':
                self.MaskFlag = True                
        self.imageSize = imageSize
        self.MaxShift = MaxShift
        self.MaxRotate = MaxRotate
        self.RotShift = RotShift
        self.FlipFlag = FlipFlag
        self.preCropImageSize = preCropImageSize
        self.PerspectiveScale=PerspectiveScale
        self.CalculateDVF = CalculateDVF
        #print(self.MaskFlag)
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.fixed_paths)
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.        
        """
        
        fixed_path = self.fixed_paths[index]
        moving_path = self.moving_paths[index]
        
        #print(fixed_path)
        #print(moving_path)
        
        fixedImage = Image.open(fixed_path)
        movingImage = Image.open(moving_path)
        
        fixedImArr = np.asarray(fixedImage)
        movingImArr = np.asarray(movingImage)
        
        fixedImage = Image.fromarray(fixedImArr.astype(dtype=np.float32))
        movingImage = Image.fromarray(movingImArr.astype(dtype=np.float32))
        
        if self.MaskFlag:
            mask_path = self.mask_paths[index]
            maskImage = Image.open(mask_path)
            maskImArr = np.asarray(maskImage)
            maskImage = Image.fromarray(maskImArr.astype(dtype=np.float32))
            
            if not(not (self.OAR_paths)):
                OAR_path = self.OAR_paths[index]
                OARImage = Image.open(OAR_path)
                OARImArr = np.asarray(OARImage)
                OARImage = Image.fromarray(OARImArr.astype(dtype=np.float32))                
                
        
        if self.FlipFlag:       
            flip = random.random() > 0.5
            
            if flip:
                fixedImage = fixedImage.transpose(Image.FLIP_TOP_BOTTOM)
                movingImage = movingImage.transpose(Image.FLIP_TOP_BOTTOM)
                if self.MaskFlag:
                    maskImage = maskImage.transpose(Image.FLIP_TOP_BOTTOM)
                    if not(not (self.OAR_paths)):
                        OARImage = OARImage.transpose(Image.FLIP_TOP_BOTTOM)
        
        if self.PerspectiveScale > 0:
            fixedImage = self.RandomPerspective(fixedImage,distortion_scale=self.PerspectiveScale)
            movingImage = self.RandomPerspective(movingImage,distortion_scale=self.PerspectiveScale)
            if self.MaskFlag:
                maskImage = self.RandomPerspective(maskImage,distortion_scale=self.PerspectiveScale)
                if not(not (self.OAR_paths)):
                    OARImage = self.RandomPerspective(OARImage,distortion_scale=self.PerspectiveScale)
        
        w,h = fixedImage.size
        #print(w)
        #print(h)
        if abs(self.MaxRotate) > 0:
            RotVal = random.uniform(-self.MaxRotate,self.MaxRotate)  
        else:
            RotVal = 0
        if abs(self.RotShift) > 0:
            RotShift = random.uniform(-self.RotShift,self.RotShift)  
        else:
            RotShift = 0
        
        if (abs(self.MaxRotate) > 0) or (abs(self.RotShift)>0):
            #fixedImage = fixedImage.rotate(RotVal,resample=Image.Resampling.BILINEAR)
            #movingImage = movingImage.rotate(RotVal+RotShift,resample=Image.Resampling.BILINEAR)
            fixedImage = fixedImage.rotate(RotVal,resample=Image.Resampling.BICUBIC)
            movingImage = movingImage.rotate(RotVal+RotShift,resample=Image.Resampling.BICUBIC)
            if self.MaskFlag:
                maskImage = maskImage.rotate(RotVal+RotShift,resample=Image.Resampling.BICUBIC)
                if not(not (self.OAR_paths)):
                    OARImage = OARImage.rotate(RotVal+RotShift,resample=Image.Resampling.BICUBIC)
            
        #Random Cropping
        if (w > self.preCropImageSize) or (h > self.preCropImageSize):
            #print('Difference In input and desired image size. Cropping')
            startX = random.randrange(0, (w - self.preCropImageSize-1), 1)  
            startY = random.randrange(0, (h - self.preCropImageSize-1), 1) 
    
            endX = startX + self.preCropImageSize
            if endX >= w:
                endX = w-1
                startX = endX-self.preCropImageSize
            endY = startY + self.preCropImageSize
            if endY >= h:
                endY = h-1
                startY = endY-self.preCropImageSize
            fixedImage = fixedImage.crop((startX, startY, startX + self.preCropImageSize, startY + self.preCropImageSize))
            
            #Slightly offset X and Y cropping between fixed and moving image
            if abs(self.MaxShift) > 0:
                XShift = random.randrange(-self.MaxShift,self.MaxShift,1)
                YShift = random.randrange(-self.MaxShift,self.MaxShift,1)
            else:
                XShift = 0
                YShift = 0
            startX = startX + XShift 
            startY = startY + YShift 
            
            endX = startX + self.preCropImageSize
            if endX >= w:
                endX = w-1
                startX = endX-self.preCropImageSize
            endY = startY + self.preCropImageSize
            if endY >= h:
                endY = h-1
                startY = endY-self.preCropImageSize    
                
            movingImage = movingImage.crop((startX, startY, startX + self.preCropImageSize,startY + self.preCropImageSize)) 
            if self.MaskFlag:
                maskImage = maskImage.crop((startX, startY, startX + self.preCropImageSize,startY + self.preCropImageSize)) 
                if not(not (self.OAR_paths)):
                    OARImage = OARImage.crop((startX, startY, startX + self.preCropImageSize,startY + self.preCropImageSize)) 
        
        elif abs(self.MaxShift) > 0:
            #print('Shifting Image')
            XShift = random.randrange(-self.MaxShift,self.MaxShift,1)
            YShift = random.randrange(-self.MaxShift,self.MaxShift,1)
            movingImage = movingImage.rotate(angle=0,translate=[XShift,YShift],expand=False)
            if self.MaskFlag:
                maskImage = maskImage.rotate(angle=0,translate=[XShift,YShift],expand=False)
                if not(not (self.OAR_paths)):
                    OARImage = OARImage.rotate(angle=0,translate=[XShift,YShift],expand=False)

        #TForm = self.GetTransforms()
        
        #fixed_tensor = TForm(fixedImage)
        #moving_tensor = TForm(movingImage)
        
        fixedImage = fixedImage.resize((self.imageSize,self.imageSize))
        movingImage = movingImage.resize((self.imageSize,self.imageSize))
        if self.MaskFlag:
            maskImage = maskImage.resize((self.imageSize,self.imageSize))    
            if not(not (self.OAR_paths)):
                OARImage = OARImage.resize((self.imageSize,self.imageSize))
        
        fixedArray = np.asarray(fixedImage)
        fixedArray = (fixedArray - np.min(fixedArray)) / (np.max(fixedArray) - np.min(fixedArray))
        movingArray = np.asarray(movingImage)
        movingArray = (movingArray - np.min(movingArray)) / (np.max(movingArray) - np.min(movingArray))
        if self.MaskFlag:
            maskArray = np.asarray(maskImage)
            maskArray = (maskArray - np.min(maskArray)) / (np.max(maskArray) - np.min(maskArray))
            if not(not (self.OAR_paths)):
                OARArray = np.asarray(OARImage)
                OARArray = (OARArray - np.min(OARArray)) / (np.max(OARArray) - np.min(OARArray))                

        fixed_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
        fixed_array[0, :, :] = np.asarray(fixedArray)
        moving_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
        moving_array[0, :, :] = np.asarray(movingArray)
        if self.MaskFlag:
            mask_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
            mask_array[0, :, :] = np.asarray(maskArray)     
            if not(not (self.OAR_paths)):
                OAR_array = np.zeros((1,self.imageSize, self.imageSize), dtype=np.float32)
                OAR_array[0, :, :] = np.asarray(OARArray)            

        target_flowArr = np.zeros((2, self.imageSize, self.imageSize), dtype=np.float32)

        if self.CalculateDVF:
            v, u = optical_flow_tvl1(fixed_array[0, :, :], moving_array[0, :, :],prefilter=True)
            target_flowArr[0, :, :] = -np.asarray(u)
            target_flowArr[1, :, :] = -np.asarray(v)

        fixed_tensor = torch.from_numpy(fixed_array)
        moving_tensor = torch.from_numpy(moving_array)
        
        target_flow = torch.from_numpy(target_flowArr)
        
        if self.MaskFlag:
            mask_tensor = torch.from_numpy(mask_array)

            if not (self.OAR_paths):
                return {'fixed': fixed_tensor, 'moving': moving_tensor, 'moving_mask':mask_tensor,
                    'target_flow':target_flow,'A_paths': fixed_path, 'B_paths': moving_path}
            else:
                OAR_tensor = torch.from_numpy(OAR_array)
                return {'fixed': fixed_tensor, 'moving': moving_tensor, 'moving_mask':mask_tensor,
                    'moving_OAR':OAR_tensor,'target_flow':target_flow,'A_paths': fixed_path, 'B_paths': moving_path}
 
        else:
        #print(fixed_tensor.shape)
        #print(moving_tensor.shape)
        
            return {'fixed': fixed_tensor, 'moving': moving_tensor, 'target_flow':target_flow,'A_paths': fixed_path, 'B_paths': moving_path}
    
    
    def GetTransforms(self):
        transform_list = []
        
        #transform_list.append(transforms.RandomHorizontalFlip())
        
        transform_list += [transforms.ToTensor()]
        #transform_list += [transforms.Normalize((0.5,), (0.5,))]
                
        return transforms.Compose(transform_list)
        
    def RandomPerspective(self, image, distortion_scale=0.5, p=0.5, interpolation=Image.BILINEAR, fill=0):
        width,height = image.size
        if random.random() < p:
            startpoints, endpoints = self.GetPespParams(width, height, distortion_scale)
            return transforms.functional.perspective(image,startpoints, endpoints, interpolation, fill)
        else:
            return image
        

    def GetPespParams(self,width: int, height: int, distortion_scale: float) -> tuple[list[list[int]], list[list[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints
    
def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = CustomImageDataset(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        #return min(len(self.dataset), self.opt.max_dataset_size)
        return len(self.dataset)

    def __iter__(self):
        
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
    