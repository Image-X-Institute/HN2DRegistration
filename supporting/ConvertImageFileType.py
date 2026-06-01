# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:35:27 2024

Yes i know shut up. 

@author: mgar5380
"""
from sys import argv
import SimpleITK as sitk

def ConvertImageFileType(InputFile,OutputFile):
    
    Img = sitk.ReadImage(InputFile)
    sitk.WriteImage(Img,OutputFile)

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    ConvertImageFileType(*argv[1:])