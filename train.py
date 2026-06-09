# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 12:36:46 2025

@author: mgar5380
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import piqa
import time
import torch
import pickle
import numpy as np
#import SimpleITK as sitk
from pathlib import Path
import torch.optim as optim
from torchsummary import summary
from matplotlib import pyplot as plt
from supporting import losses
from supporting.model import VxmDense
from supporting.options import TrainOptions
#from supporting.visualizer import Visualizer

from supporting.dataset import CustomImageDataset,CustomImageDVFDataset


if __name__ == '__main__':
    torch.cuda.empty_cache()
    opt = TrainOptions().parse()   # get training options
    
    #dataset = create_dataset(opt.fixedDir,opt.movingDir,imageSize=opt.imageSize,rescale=opt.rescale)  # create a dataset given opt.dataset_mode and other options
    if opt.Supervised:
        dataset = CustomImageDVFDataset(opt.fixedDir,opt.movingDir,DVFDir=opt.DVFDir,
                                    imageSize=opt.imageSize,
                                    preCropImageSize=opt.preCropImageSize,MaxRotate=opt.MaxRotate,
                                    FlipFlag=opt.FlipFlag,datasetSize=opt.max_dataset_size)
    
    else:
        dataset = CustomImageDataset(opt.fixedDir,opt.movingDir,imageSize=opt.imageSize,
                                     preCropImageSize=opt.preCropImageSize,
                                     MaxShift=opt.MaxShift,MaxRotate=opt.MaxRotate,RotShift=opt.RotShift,
                                     PerspectiveScale=opt.PerspectiveScale,FlipFlag=opt.FlipFlag)


    
    #generator1 = torch.Generator().manual_seed(42)
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    split = [int(round(len(dataset) * opt.TrainingValSplit)), int(round(len(dataset) * (1.0-opt.TrainingValSplit)))]
    trainset, valset = torch.utils.data.dataset.random_split(dataset, split)#,generator=generator1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=True)    
    
    print('The number of training/validation images = {},{}'.format(len(trainset),len(valset)))
    
    if opt.networkType == 'voxelmorph':
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        
        model = VxmDense(
            inshape=[opt.imageSize,opt.imageSize],
            nb_unet_features=[enc_nf, dec_nf]
            )
    
        name='VXMUnet'
    else:
        raise NameError('Value {} for networkType unrecognised'.format(opt.networkType))
        
    if int(opt.gpu_ids[0]) < 0:
        device=torch.device("cpu")
    else:
        device = torch.device(("cuda:"+opt.gpu_ids) if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    
    
    if opt.continue_train:
        save_filename = '%s_net_%s.pth' % (opt.epoch_count, name)
        networkWeightsFile = os.path.join(opt.checkpoints_dir,opt.name,save_filename)
        model.load_state_dict(torch.load(networkWeightsFile, map_location=device))
    
    train_tmp = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    data = next(iter(train_tmp))
    input_size_image = data['fixed'].shape[1:]
    #print(input_size_image)
    if opt.verbose:
        print(summary(model, input_size = [input_size_image, input_size_image]))
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.fineTune:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",patience=5,
                                            threshold=0.0005)
    
    #Select loss function
    if opt.LossType == 'MSE':
        MainLoss = losses.MSE()
    elif opt.LossType == 'NCC':
        MainLoss = losses.NCC(device=device,win=[int(opt.imageSize/6.0)]*2)
    elif opt.LossType == 'MI':
        MainLoss = losses.MutualInformation()
    elif opt.LossType == 'L1':
        MainLoss = losses.L1()        
    else:
        raise NameError('Value {} for LossType unrecognised'.format(opt.LossType))
    GradLoss = losses.Grad()
    
    #if opt.SSIMParam > 0:
    SSIMLoss = piqa.ssim.SSIM(n_channels=1).to(device)
    
    AffineLoss = losses.Affine()
    
    #model = create_model(opt)      # create a model given opt.model and other options
    #model.setup(opt)               # regular setup: load and print networks; create schedulers

    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #total_iters = 0                # the total number of training iterations
    
    model_start_time = time.time()
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    nvec = 20  # Number of vectors to be displayed along each image dimension
    
    
    if not Path(os.path.join(opt.checkpoints_dir,opt.name)).exists():
        Path(os.path.join(opt.checkpoints_dir,opt.name)).mkdir(parents=True)
    
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
            
    LastWrittenEpoch = opt.epoch_count   
    
    for epoch in range(opt.epoch_count+1, opt.niter + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        #iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        train_loss = 0.0

        for i, data in enumerate(trainloader,0):  # inner loop within one epoch
            #iter_start_time = time.time()  # timer for computation per iteration
            #if total_iters % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            #total_iters += opt.batch_size
           #epoch_iter += opt.batch_size
            #
            optimizer.zero_grad()
            
            if opt.Supervised:
                fixed_proj, moving_proj,target_flow = data['fixed'].to(device), data['moving'].to(
                device), data['target_flow'].to(device)
            else:
                fixed_proj, moving_proj = data['fixed'].to(device), data['moving'].to(
                device)
            #print(moving_proj.shape)
            #print(fixed_proj.shape)
            
            if opt.networkType == 'voxelmorph':
                warped_proj,estFlow = model.forward(moving_proj, fixed_proj)#,registration=True)
            elif opt.networkType == 'transmorph':
                warped_proj,estFlow = model.forward(torch.cat((moving_proj, fixed_proj), dim=1))
                
            loss2 = GradLoss.loss(estFlow) 
           
            ALoss = AffineLoss.loss(estFlow) 
           
            if opt.Supervised:
                loss1 = MainLoss.loss(target_flow,estFlow)
                lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2 + opt.AffineLoss*ALoss
            else:
                #if opt.LossType == 'NCC':
                #    loss1 = MainLoss.loss(fixed_proj,warped_proj,device)
                #else:
                loss1 = MainLoss.loss(fixed_proj,warped_proj)
                loss2 = GradLoss.loss(estFlow)
                loss3 = 1.0 - SSIMLoss(fixed_proj,warped_proj)
                
                lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2 + opt.SSIMParam*loss3 + opt.AffineParam*ALoss
            
            #lossLoop = MSELoss.loss(fixed_proj,warped_proj)
            
                # if opt.GradParam == 0:
                #     #lossLoop = opt.LossParam*loss1
                #     if opt.SSIMParam > 0:   
                #         loss3 = 1.0 - SSIMLoss(fixed_proj,warped_proj)
                #         lossLoop = opt.LossParam*loss1 + opt.SSIMParam*loss3
                #     else:
                #         lossLoop = opt.LossParam*loss1
                # else:
                # #loss = opt.MSEParam*MSELoss(fixed_proj,warped_proj) + opt.GradParam*GradLoss(estFlow)
                #     if opt.SSIMParam > 0:    
                #         loss3 = 1.0 - SSIMLoss(fixed_proj,warped_proj)
                #         lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2 + opt.SSIMParam*loss3
                #     else:
                #         loss1 =  MainLoss.loss(fixed_proj,warped_proj)
                #         lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2
               
           # lossLoop = opt.MSEParam*loss1.item() + opt.GradParam*loss2.item()
            #opt.MSEParam*MSELoss(fixed_proj,warped_proj) + opt.GradParam*GradLoss(estFlow)
            train_loss += lossLoop.item()
            lossLoop.backward()
            #train_loss += loss.item()
            optimizer.step()
            
            if i % opt.print_freq == 0:
                print('Epoch {}: Iteration {}'.format(epoch,i))
            
        if opt.fineTune:
            scheduler.step(train_loss)
            
        # test and print every epoch
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for j, valdata in enumerate(valloader, 0):
                if opt.Supervised:
                    fixed_proj, moving_proj,target_flow = data['fixed'].to(device), data['moving'].to(
                    device), data['target_flow'].to(device)
                else:
                    fixed_proj, moving_proj = data['fixed'].to(device), data['moving'].to(
                    device)
                if opt.networkType == 'voxelmorph':
                    warped_proj,estFlow = model.forward(source=moving_proj,target=fixed_proj,registration=True)
                elif opt.networkType == 'transmorph':
                    warped_proj,estFlow = model.forward(torch.cat((moving_proj, fixed_proj), dim=1))
                
                #warped_proj,estFlow = model.forward(fixed_proj, moving_proj)#,registration=True)
                loss1 = MainLoss.loss(fixed_proj,warped_proj)
                loss2 = GradLoss.loss(estFlow)
                loss3 = 1.0 - SSIMLoss(fixed_proj,warped_proj)
                ALoss = AffineLoss.loss(estFlow) 
                
                lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2 + opt.SSIMParam*loss3 + opt.AffineParam*ALoss
                #if opt.Supervised:
                #    warped_projReal = model.transformer(moving_proj,target_flow)
                ##    loss1 = MainLoss.loss(target_flow,estFlow)
                #   lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2
                #else:
                    #if opt.LossType == 'NCC':
                    #    loss1 = MainLoss.loss(fixed_proj,warped_proj,device)
                    #else:
                #    loss1 =  MainLoss.loss(fixed_proj,warped_proj)
                    #loss2 = GradLoss.loss(estFlow)
                #    if opt.SSIMParam > 0:    
                #        loss3 = 1.0 - SSIMLoss(fixed_proj,warped_proj)
                #        lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2 + opt.SSIMParam*loss3
                #    else:
                #        lossLoop = opt.LossParam*loss1 + opt.GradParam*loss2
                #lossLoop = MSELoss.loss(fixed_proj,warped_proj)
                val_loss += lossLoop.item()
                
            #print(moving_proj[0,:,:,:].shape)
            #print(fixed_proj[0,:,:,:].shape)
                
            #print(estFlow.shape)
            #print(target_flow.shape)
                
            plt.figure()
            plt.imshow(np.squeeze(moving_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
            plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_Moving.png' % epoch))
            plt.close()
            
            plt.figure()
            plt.imshow(np.squeeze(fixed_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')  
            plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_Fixed.png' % epoch))              
            plt.close()

            plt.figure()
            plt.imshow(np.squeeze(warped_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
            plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_Warped.png' % epoch))
            plt.close()

            # if opt.Supervised:
            #     plt.figure()
            #     plt.imshow(np.squeeze(warped_projReal[0,:,:,:].detach().cpu().numpy()), cmap='gray')
            #     plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_WarpedReal.png' % epoch))
            #     plt.close()
            
            plt.figure()
            plt.imshow(abs(np.squeeze(warped_proj[0,:,:,:].detach().cpu().numpy())-np.squeeze(fixed_proj[0,:,:,:].detach().cpu().numpy())), cmap='gray')   
            plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_Diff.png' % epoch))            
            plt.close()

            # if opt.Supervised:
            #     plt.figure()
            #     plt.imshow(abs(np.squeeze(warped_projReal[0,:,:,:].detach().cpu().numpy())-np.squeeze(fixed_proj[0,:,:,:].detach().cpu().numpy())), cmap='gray')   
            #     plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'%d_RealDiff.png' % epoch))            
            #     plt.close()
            
            nl, nc = np.squeeze(warped_proj[0,:,:,:].detach().cpu().numpy()).shape
            step = max(nl // nvec, nc // nvec)
            
            y, x = np.mgrid[:nl:step, :nc:step]
            u = -np.squeeze(estFlow[0,0,:,:].cpu().detach().numpy())
            v = -np.squeeze(estFlow[0,1,:,:].cpu().detach().numpy())
            u_ = u[::step, ::step]
            v_ = v[::step, ::step]

            if opt.Supervised:
                u2 = -np.squeeze(target_flow[0,0,:,:].cpu().detach().numpy())
                v2 = -np.squeeze(target_flow[0,1,:,:].cpu().detach().numpy())
                u2_ = u2[::step, ::step]
                v2_ = v2[::step, ::step]
            
            plt.figure()
            plt.imshow(np.squeeze(moving_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
            plt.quiver(x, y, u_, v_, color='r', units='dots', angles='xy', scale_units='xy')
            #plt.patch.set_visible(False)
            plt.axis('off')              
            plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'MovingWithEstDVF_%d.png' % epoch))
            plt.close()
            
            if opt.Supervised:
                plt.figure()
                plt.imshow(np.squeeze(moving_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
                plt.quiver(x, y, u2_, v2_, color='r', units='dots', angles='xy', scale_units='xy')
                #plt.patch.set_visible(False)
                plt.axis('off')              
                plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'MovingWithRealDVF_%d.png' % epoch))
                plt.close() 
                
                plt.figure()
                plt.imshow(np.squeeze(moving_proj[0,:,:,:].detach().cpu().numpy()), cmap='gray')
                plt.quiver(x, y, (u_-u2_), (v_-v2_), color='r', units='dots', angles='xy', scale_units='xy')
                #plt.patch.set_visible(False)
                plt.axis('off')              
                plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'MovingWithDVFDiff_%d.png' % epoch))
                plt.close()            
 
            
        Epoch_elapsed = (time.time()- epoch_start_time) / 3600
        Epoch_hours = np.floor(Epoch_elapsed)
        Epoch_minutes = (Epoch_elapsed - Epoch_hours) * 60

        Total_elapsed = (time.time()- model_start_time) / 3600
        Total_hours = np.floor(Total_elapsed)
        Total_minutes = (Total_elapsed - Total_hours) * 60
            
        message = 'Epoch: %d | train loss: %.5f | val loss: %.5f | total time: %d hours %d minutes' %(epoch, 
                                                                                                      train_loss / len(trainset),
                                                                                                      val_loss / len(valset),
                                                                                                      Total_hours,
                                                                                                      Total_minutes)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message        
    
        train_losses.append(train_loss / len(trainset))
        val_losses.append(val_loss / len(valset))
        
        #if (val_loss < min_val_loss) | (epoch-LastWrittenEpoch > 20):
        LastWrittenEpoch = epoch
        save_filename = '%s_net_%s.pth' % (epoch, name)
        networkPath = os.path.join(opt.checkpoints_dir,opt.name,save_filename)
        torch.save(model.state_dict(), networkPath)
        min_val_loss = val_loss
        
        # plot training
        plt.figure()
        #plt.title(expt_description)
        plt.plot(np.array(range(opt.epoch_count+1, epoch + 1)), np.array(train_losses), 'b')
        plt.plot(np.array(range(opt.epoch_count+1, epoch + 1)), np.array(val_losses), 'r')
        plt.legend(['Train', 'Validation'],loc='upper right')
        plt.ylabel('Loss')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if Total_minutes > 30:
            Total_hours += 1
        plt.xlabel('Epochs' + ' (' + str(int(Total_hours)) + ' hours)')
        plt.savefig(os.path.join(opt.checkpoints_dir,opt.name,'Losses.png'))
        plt.close()
        
    print('Finished training')
    torch.cuda.empty_cache()
    
    with open(os.path.join(opt.checkpoints_dir,opt.name,'Losses.pkl'), 'wb') as f:
        pickle.dump((train_losses, val_losses), f)
    
            #model.set_input(data)         # unpack data from dataset and apply preprocessing
            #model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     OverwriteSaveFlag = 1
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result,OverwriteSaveFlag)
                
            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)
            
            # iter_data_time = time.time()
            
        #if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #    print('saving the model at the end of epoch %d' % (epoch))
        #    model.save_networks(epoch,save_dir=opt.checkpoints_dir)
            
        #print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        #model.update_learning_rate()  

