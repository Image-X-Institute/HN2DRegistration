# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 12:52:39 2025

@author: mgar5380
"""
import os
import torch
import argparse
from supporting import util

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--fixedDir', required=True, help='path to fixed images')
        parser.add_argument('--movingDir', type=str, default='',help='path to moving images. If left empty movingDir=fixedDir')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--verbose',action='store_true',help='Whether to make network data visible')

        #dataset parameters
        parser.add_argument('--imageSize', type=int, default=512, help='Image size that the input images will be cropped and rescaled to.')
        parser.add_argument('--preCropImageSize', type=int, default=512, help='Size the input image will be cropped to.')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        
        #Model parameters
        parser.add_argument('--networkType',type=str,default='voxelmorph',help='The type of network used for training/testing [voxelmorph|transmorph]')
        parser.add_argument('--configFile',type=str,default='',help='The location of the network cofig file.')
        self.initialized = True
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        if opt.verbose:
            print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if self.isTrain:
            file_name = os.path.join(expr_dir, 'opt_train.txt')
        else:
            file_name = os.path.join(expr_dir, 'opt_test.txt')
        
        
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        #if opt.suffix:
        #    suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #    opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
    
class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
       # visdom and HTML visualization parameters
        #parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        #parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        #parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        #parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
       # parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
       # parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
       # parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        #parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        #parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        #parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        #parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--LossParam',type=float,default=1.0,help='Scaling factor for Main loss')
        parser.add_argument('--GradParam',type=float,default=0.1,help='Scaling factor for Grad loss')
        parser.add_argument('--SSIMParam',type=float,default=0.0,help='Scaling Factor for SSIM loss')
        parser.add_argument('--LossType',type=str,default='MSE',help='Loss Type (options: MSE|NCC|MI|L1)')
        parser.add_argument('--AffineParam',type=float,default=0,help='Saclign factor for affine loss')
        parser.add_argument('--TrainingValSplit',type=float,default=0.9,help='Training/validation split for training data')
        #parser.add_argument('--LossDomain',type=str,default='DVF',help='Whether to calculate the loss function in the image or DVF domain {image|DVF}. Default is DVF')
        parser.add_argument('--fineTune',action='store_true',help='If true then training code used to finetune a network, not to train from scratch')
        #Dataset options
        parser.add_argument('--Supervised',action='store_true',help='If true the network will be trained in a supervised manner with ground truth DVFs')
        parser.add_argument('--MaxShift', type=float, default=60.0, help='Maximum difference in pixel values between fixed and moving images (for data augmentation)')
        parser.add_argument('--MaxRotate',type=float,default=10.0,help='Maximum rotation for both fixed and moving images.')
        parser.add_argument('--RotShift',type=float,default=3.0,help='Maximum rotation difference in degrees between fixed and moving images (for data augmentation)')
        parser.add_argument('--PerspectiveScale',type=float,default=0.5,help='Magnitude of the pespective distortion. Number is between 0(min) and 1(max).')
        parser.add_argument('--DVFDir',type=str,default='',help='Location of the ground truth DVFs')
        parser.add_argument('--FlipFlag',action='store_true',help='Flag for whether the data is to be flipped during data augmentation')
                            
        self.isTrain = True
        return parser
        
class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--maskDir', type=str, default='', help='directory of masks to warp during testing phase.If empty only warped testing images will be saved')
        parser.add_argument('--OARDir', type=str, default='', help='directory of OAR to warp during testing phase.If empty only warped testing images will be saved')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--epoch',type=int,help='Epoch To load')
        #Add random rigid deformation for moving images. 
        parser.add_argument('--testingMotion',action='store_true',help='If true, synthetic motion will be added to the moving images and masks')
        parser.add_argument('--MovingShift',type=float,default=60.0,help='Maximum difference the moving image will be shifted by (in pixels)')
        parser.add_argument('--RotShift',type=float,default=3.0,help='Maximum difference the moving image will be shifted by (in pixels)')
        self.isTrain = False
        return parser
    
