# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:13:32 2026

@author: mgar5380
"""
import os
import glob
import numpy as np
from sys import argv
from pathlib import Path

def ConvertBinToNumpy(InputDir,OutputDir='',imSize=[1024,768],OverwriteFlag=False):
    
    if not OutputDir:
        OutputDir = InputDir
    
    if not Path(OutputDir).exists():
        Path(OutputDir).mkdir(parents=True)
    
    DvfFileList = glob.glob(os.path.join(InputDir,'*DVFu.bin'))
    
    for f in DvfFileList:
        
        fileNameLong = os.path.basename(f)
        
        StrSplit = fileNameLong.split('_')
        
        fileNameShort = ''
        for i in range(0,len(StrSplit)-1):
            fileNameShort = fileNameShort + StrSplit[i] + '_'
        
        fileNameShort = fileNameShort[:-1] + '.npy'
            
        newFileName = os.path.join(OutputDir,fileNameShort)
        
        if (not Path(newFileName).exists()) or OverwriteFlag:
            
            DVFu = np.fromfile(f,dtype='single')
            DVFu = np.reshape(DVFu,imSize,order='F')
            
            DVFv = np.fromfile(os.path.join(InputDir,fileNameShort[:-4]+'_DVFv.bin'),dtype='single')
            DVFv = np.reshape(DVFv,imSize,order='F')
            
            DVFNew = np.zeros((DVFu.shape[0],DVFu.shape[1],2),dtype='single')
            DVFNew[:,:,0] = DVFu
            DVFNew[:,:,1] = DVFv
            
            np.save(os.path.join(OutputDir,fileNameShort),DVFNew)


InputDir = 'Y:\\PRJ-RPL\\2RESEARCH\\2_ProjectData\\RemoveTheMask\\CTData\\HNSCC\\movingDVFBin'
OutputDir = 'Y:\\PRJ-RPL\\2RESEARCH\\2_ProjectData\\RemoveTheMask\\CTData\\HNSCC\\movingDVFnpy'

ConvertBinToNumpy(InputDir,OutputDir)

#if __name__ == '__main__':
    # Map command line arguments to function arguments.
#    ConvertBinToNumpy(*argv[1:])