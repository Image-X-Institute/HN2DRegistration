# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:07:07 2025

@author: mgar5380
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import timeit
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from supporting.model import VxmDense
from supporting.warping import WarpImage
from supporting.options import TestOptions
from supporting.util import tensor2im,save_image
from supporting.dataset import CustomImageDataset,CustomImageDVFDataset

def ReadMaskImage(maskDir,i,imageSize):
    
    ImName = 'Img_{:04g}.png'.format(i+1)

    inputImageFile = os.path.join(maskDir,ImName)
    
    if not Path(inputImageFile).exists():
        ImName = 'im_{:04g}.png'.format(i+1)
        inputImageFile = os.path.join(maskDir,ImName)
    
    fixedImage = Image.open(inputImageFile)
    
    fixedImArr = np.asarray(fixedImage)
    
    fixedImage = Image.fromarray(fixedImArr.astype(dtype=np.float32))
    
    #DVFUImage.resize((self.imageSize,self.imageSize))
    
    fixedArray = np.asarray(fixedImage.resize((imageSize,imageSize)))
    fixedArray = (fixedArray - np.min(fixedArray)) / (np.max(fixedArray) - np.min(fixedArray))

    fixed_array = np.zeros((1,1,imageSize, imageSize), dtype=np.float32)
    fixed_array[0,0, :, :] = np.asarray(fixedArray)    

    maskTensor = torch.from_numpy(fixed_array)

    return maskTensor

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    #Set initial conditions
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    #opt.rescale = None
    if opt.testingMotion:
        RotShift = opt.RotShift
        MaxShift = opt.MovingShift
        print('Simulating Motion')
    else:
        RotShift = 0
        MaxShift = 0
        
    MaxRotate = 0
    #opt.RotShift = 0
    PerspectiveScale = 0
    name='VXMUnet'
    mask_str = 'mask'
    FlipFlag=False
    
    
    #Create dataset
    #dataset = CustomImageDVFDataset(opt.fixedDir,opt.movingDir,imageSize=opt.imageSize,
    #                                FlipFlag=FlipFlag,MaxRotate=opt.MaxRotate,phase='test')
    # dataset = CustomImageDataset(opt.fixedDir,opt.movingDir,imageSize=opt.imageSize,rescale=opt.rescale,
    #                              MaxShift=opt.MaxShift,MaxRotate=opt.MaxRotate,RotShift=opt.RotShift,
    #                              FlipFlag=FlipFlag,PerspectiveScale=0.0)
    
    dataset = CustomImageDataset(opt.fixedDir,opt.movingDir,imageSize=opt.imageSize,
                                 maskDir=opt.maskDir,phase='test',
                                 preCropImageSize=opt.preCropImageSize,
                                 MaxShift=MaxShift,MaxRotate=MaxRotate,RotShift=RotShift,
                                 PerspectiveScale=PerspectiveScale,FlipFlag=FlipFlag,OARDir=opt.OARDir)

    
    testloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)
    
    #Load model
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    
    model = VxmDense(
        inshape=[opt.imageSize,opt.imageSize],
        nb_unet_features=[enc_nf, dec_nf]
        )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Load model weights
    save_filename = '%s_net_%s.pth' % (opt.epoch, name)
    networkWeightsFile = os.path.join(opt.checkpoints_dir,opt.name,save_filename)
    model.load_state_dict(torch.load(networkWeightsFile, map_location=device))
    
    #Set directory for outputresults
    outputDir = Path(opt.results_dir,opt.name,'test_{:04g}'.format(opt.epoch))
    if not Path(outputDir).exists():
        Path(outputDir).mkdir(parents=True)
    
    times = []
    
    if opt.eval:
        model.eval()
    for i, data in enumerate(testloader,0):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        start_time = timeit.default_timer()
        
        #print(moving_proj.shape)
    
        #print(fixed_proj.shape)
        if not(not opt.maskDir):
            
            if not opt.OARDir:
            
                moving_proj, fixed_proj, mask_proj = data['moving'].to(device), data['fixed'].to(
                device), data['moving_mask'].to(device)
                
                #mask_proj = ReadMaskImage(opt.maskDir,i,opt.imageSize).to(device)
                #print(mask_proj.shape)
                warped_proj,estFlow,warped_mask = model.forward(moving_proj, fixed_proj,
                                                                registration=True,
                                                                mask=mask_proj,MaskFlag=True)
                                                                
            else:
                moving_proj, fixed_proj, mask_proj, OAR_proj = data['moving'].to(device), data['fixed'].to(
                device), data['moving_mask'].to(device), data['moving_OAR'].to(device)
                
                warped_proj,estFlow,warped_mask = model.forward(moving_proj, fixed_proj,
                                                                registration=True,
                                                                mask=mask_proj,MaskFlag=True)
                                                                
                warped_OAR = model.forward(moving_proj, fixed_proj,
                                                                registration=True,
                                                                mask=OAR_proj,MaskFlag=True)[2]           
        else:
            
            moving_proj, fixed_proj = data['moving'].to(device), data['fixed'].to(
            device)
            
            warped_proj,estFlow = model.forward(moving_proj, fixed_proj,registration=True)
   
        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)
        
        image_name = 'WarpedImage%04d.png' % (i)
        image_path = os.path.join(outputDir,image_name)
           
        nvec = 20  # Number of vectors to be displayed along each image dimension
        nl, nc = np.squeeze(warped_proj[0,:,:,:].detach().cpu().numpy()).shape
        step = max(nl // nvec, nc // nvec)
        
        y, x = np.mgrid[:nl:step, :nc:step]
        u = -np.squeeze(estFlow[0,0,:,:].cpu().detach().numpy())
        v = -np.squeeze(estFlow[0,1,:,:].cpu().detach().numpy())
        u_ = u[::step, ::step]
        v_ = v[::step, ::step]

        movingIm = tensor2im(moving_proj, imtype=np.uint8)
        fixedIm = tensor2im(fixed_proj, imtype=np.uint8)
        warpedIm = tensor2im(warped_proj, imtype=np.uint8)

        movingImage = Image.fromarray(movingIm)
        fixedImage = Image.fromarray(fixedIm)
        warpedImage = Image.fromarray(warpedIm)

        movingImage.save(os.path.join(outputDir,'Moving_%d.png' % (i+1)))
        fixedImage.save(os.path.join(outputDir,'Fixed_%d.png' % (i+1)))
        warpedImage.save(os.path.join(outputDir,'Warped_%d.png' % (i+1)))

        plt.figure()
        plt.imshow(np.squeeze(moving_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
        plt.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy')
        #plt.patch.set_visible(False)
        plt.axis('off')              
        plt.savefig(os.path.join(outputDir,'MovingWithDVF_%d.png' % (i+1)))
        plt.close()
        
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image...' % (i))
            
        #Warp the Corresponding Mask Image
        if not(not opt.maskDir):
            #print(estFlow.shape)
            #warpedMask = WarpImage(estFlow[0,:,:,:].cpu().detach().numpy(),opt.maskDir,i)
            #warpedProj = WarpImage(estFlow[0,:,:,:].cpu().detach().numpy(),opt.movingDir,i)
            #print(warpedMask.shape)
            #image_name = 'Warped'+mask_str+'%04d.png' % (i)
            #image_path = os.path.join(outputDir,image_name)
            
            maskIm = tensor2im(warped_mask, imtype=np.uint8)
            maskImage = Image.fromarray(maskIm)
            maskImage.save(os.path.join(outputDir,'WarpedMask_%d.png' % (i+1)))
 
            maskIm = tensor2im(mask_proj, imtype=np.uint8)
            maskImage = Image.fromarray(maskIm)
            maskImage.save(os.path.join(outputDir,'UnWarpedMask_%d.png' % (i+1)))           
 
            if not(not opt.OARDir):
                OARIm = tensor2im(warped_OAR, imtype=np.uint8)
                OARImage = Image.fromarray(OARIm)
                OARImage.save(os.path.join(outputDir,'WarpedOAR_%d.png' % (i+1)))
     
                OARIm = tensor2im(OAR_proj, imtype=np.uint8)
                OARImage = Image.fromarray(OARIm)
                OARImage.save(os.path.join(outputDir,'UnWarpedOAR_%d.png' % (i+1))) 
            # plt.figure()
            # plt.imshow(np.squeeze(warped_mask[0,:,:,:].detach().cpu().numpy()), cmap='gray')
            # #plt.patch.set_visible(False)
            # plt.axis('off')                 
            # plt.savefig(os.path.join(opt.results_dir,opt.name,'WarpedMask_%d.png' % i))
            # plt.close()
            
            #save_image(warpedMask.astype(np.uint8), image_path)
            
    with open(os.path.join(outputDir,'times.txt'), 'w') as f:
        for line in times:
            f.write(f"{line}\n")