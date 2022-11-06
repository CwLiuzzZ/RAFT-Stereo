from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo import RAFTStereo

from evaluate_stereo import *
import core.stereo_datasets as datasets
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import myself.toolkit as toolkit
from myself import dataset
from myself.seed_growth import seed_growth
sys.path.append("../LoFTR_master")
from LoFTR_master.lofrt_match import LoFTR_match

# os.environ["CUDA_VISIBLE_DEVICES"]= '2,3'
os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='raft-stereo', help="name your experiment")
parser.add_argument('--data_type', default='virtualroad', help="type your dataset")
parser.add_argument('--restore_ckpt', default='models/raftstereo-eth3d.pth', help="restore checkpoint")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

# Training parameters
parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
parser.add_argument('--epochs', type=int, default=15, help="batch size used during training.")
parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--num_steps', type=int, default=50000, help="length of training schedule.")
parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

# Validation parameters
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

# Architecure choices
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', default=[0.0, 1.4], type=float, nargs='+',  help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
args = parser.parse_args()

torch.manual_seed(1234)
np.random.seed(1234)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

Path("checkpoints").mkdir(exist_ok=True, parents=True)
    
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    # assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    # valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    # assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    # assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()
    # epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]

    # metrics = {
    #     'epe': epe.mean().item(),
    #     '1px': (epe < 1).float().mean().item(),
    #     '3px': (epe < 3).float().mean().item(),
    #     '5px': (epe < 5).float().mean().item(),
    # }

    # return flow_loss, metrics
    return flow_loss, {}


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args,train_loader,TestImgLoader,model_name):
    
    #SAVE
    save_path = 'trained/{}'.format(model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    start_full_time = time.time()
    model = nn.DataParallel(RAFTStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint form {}".format(args.restore_ckpt))
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    global_batch_num = 0
    train_loss_list = []
    valid_loss_list = []
    
    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        ## training ##
        model.module.freeze_bn()
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            start_time = time.time() 
            optimizer.zero_grad()
            imgL   = sample['left']
            imgR   = sample['right']
            left_dir = sample['left_dir']
            disp_true = sample['disp'].unsqueeze(1)
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
            pad = toolkit.InputPadder(imgL.shape,16, mode = 'replicate')
            image1, image2 = pad.pad(imgL, imgR)

            mask = (disp_true > 0)
            mask.detach_()

            assert model.training
            flow_predictions = model(image1, image2, iters=args.train_iters)
            for i in range(len(flow_predictions)):
                flow_predictions[i] = -pad.unpad(flow_predictions[i])
            assert model.training
            
            disp_vis = flow_predictions[-1][0][0].detach().cpu().numpy()
            print(np.mean(disp_vis))
            toolkit.disp_vis_PLA('delete/{}.png'.format(str(batch_idx)),disp_vis,150)
            
            
            if np.random.random() < 0.05:
                # print(np.max(flow_predictions[-1][0][0].detach().cpu().numpy()))
                # print(np.min(flow_predictions[-1][0][0].detach().cpu().numpy()))
                # print(np.mean(flow_predictions[-1][0][0].detach().cpu().numpy()))
                toolkit.disp_vis('train.png',flow_predictions[-1][0][0].detach().cpu().numpy())

            loss, metrics = sequence_loss(flow_predictions, disp_true, mask)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss.item()
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(train_loader)))
        train_loss_list.append(total_train_loss/len(train_loader))
    
        ## Test ##
        model.eval()
        print('strat validating')
        D1_all_all = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            imgL   = sample['left']
            imgR   = sample['right']
            disp_true = sample['disp'].unsqueeze(1)
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
            pad = toolkit.InputPadder(imgL.shape,16, mode = 'replicate')
            image1, image2 = pad.pad(imgL, imgR)

            mask = (disp_true > 0)
            mask.detach_()
            
            with torch.no_grad():
                flow_predictions = model(image1, image2, iters=args.valid_iters)
            for i in range(len(flow_predictions)):
                flow_predictions[i] = -pad.unpad(flow_predictions[i]).squeeze(0)
            
            pred_disp = flow_predictions[-1].squeeze(1).detach().cpu().numpy()
            gt_disp = disp_true.squeeze(1).detach().cpu().numpy()
            if np.random.random() < 0.05:
                # print(np.max(flow_predictions[-1][0][0].detach().cpu().numpy()))
                # print(np.min(flow_predictions[-1][0][0].detach().cpu().numpy()))
                # print(np.mean(flow_predictions[-1][0][0].detach().cpu().numpy()))
                toolkit.disp_vis('valid.png',flow_predictions[-1][0][0].detach().cpu().numpy())
            loss, metrics = sequence_loss(flow_predictions, disp_true, mask)
            print('Iter %d val loss = %.3f' %(batch_idx, loss))
            total_test_loss += loss.item()
            
            
            wrong = np.zeros(pred_disp.shape)
            diff = abs(pred_disp-gt_disp)
            wrong[diff>3]=1
            wrong[diff>gt_disp*0.05]=1
            D1_all = np.sum(wrong)/760/1280
            D1_all_all += D1_all
            
        print('valid D1:{}. epoch:{}'.format(D1_all_all/75,epoch))
        print('epoch %d val loss = %.3f' %(epoch, total_test_loss/len(TestImgLoader)))
        valid_loss_list.append(total_test_loss/len(TestImgLoader))

            
        print('saving in {}'.format(save_path))
        np.save(os.path.join(save_path,'train_loss.npy'),np.array(train_loss_list))
        np.save(os.path.join(save_path,'valid_loss.npy'),np.array(valid_loss_list))
        len_x = np.arange(1,len(valid_loss_list)+1)
        plt.plot(len_x,np.array(train_loss_list),color='blue')
        plt.plot(len_x,np.array(valid_loss_list),color='yellow')
        plt.savefig(os.path.join(save_path,'loss.png'))
        plt.close()
        
        savefilename = os.path.join(save_path,'finetune_'+str(epoch)+'.pth')
        print('saving model in {}'.format(savefilename))
        # logging.info(f"Saving file {savefilename.absolute()}")
        torch.save(model.state_dict(), savefilename)

        
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    # print(min_epo)
    # print(min_val)
    

    print("FINISHED TRAINING")
    logger.close()

# def seed_train_middlebury(args,train_loader):
        
#     start_full_time = time.time()
#     train_loss_list = []
#     model = nn.DataParallel(RAFTStereo(args))
#     print("Parameter Count: %d" % count_parameters(model))

#     optimizer, scheduler = fetch_optimizer(args, model)
#     if args.restore_ckpt is not None:
#         assert args.restore_ckpt.endswith(".pth")
#         print("Loading checkpoint form {}".format(args.restore_ckpt))
#         checkpoint = torch.load(args.restore_ckpt)
#         model.load_state_dict(checkpoint, strict=True)

#     model.cuda()
#     model.train()
#     model.module.freeze_bn() # We keep BatchNorm frozen

#     scaler = GradScaler(enabled=args.mixed_precision)    
    
#     for epoch in range(1, args.epochs+1):
#         total_train_loss = 0
#         model.module.freeze_bn()
#         for batch_idx, sample in enumerate(train_loader):
#             start_time = time.time() 
#             imageL   = sample['left'].cuda()
#             imageR   = sample['right'].cuda()
#             imageL_dir = sample['left_dir'][0]
#             imageR_dir = sample['right_dir'][0]
#             save_dir = sample['dir'][0]
#             # disp_true = sample['disp'].unsqueeze(1).cuda()
#             print('processing {}'.format(imageL_dir))
            
#             ori_shape = (imageL.shape[2],imageL.shape[3])
#             image1 = F.interpolate(imageL, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)
#             image2 = F.interpolate(imageR, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)
#             padder = toolkit.InputPadder(image1.shape, divide = 16, mode = 'replicate') # [N,H,W]
#             image1, image2 = padder.pad(image1, image2)
            
#             ## seed and validate ##
#             model.eval()
#             resize_shape = (512,1024)
#             with torch.no_grad():
#                 mkpts0,mkpts1 = LoFTR_match([imageL_dir,imageR_dir],'indoor',resize_shape)
#                 with autocast(enabled=False):
#                     _,flow_pr,fmap1,fmap2 = model(image1, image2, iters=args.train_iters,test_mode=True)
#                 flow_pr = padder.unpad(flow_pr)
#                 raft_disp = F.interpolate(-flow_pr*ori_shape[1]/flow_pr.shape[-1], size=ori_shape, mode='bilinear', align_corners=True).squeeze().detach().cpu().numpy()
#                 ratio = int(image1.shape[-1]/fmap1.shape[-1])
#                 new_size = (ratio * fmap1.shape[-2], ratio * fmap1.shape[-1])
#                 fmap1 = F.interpolate(fmap1, size=new_size, mode='bilinear', align_corners=True) # [1,C,H,W]
#                 fmap2 = F.interpolate(fmap2, size=new_size, mode='bilinear', align_corners=True)
#                 fmap1 = padder.unpad(fmap1[0]) # [C,H,W]
#                 fmap2 = padder.unpad(fmap2[0])
#                 fmap1 = F.interpolate(fmap1.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True) # [1,C,H,W]
#                 fmap2 = F.interpolate(fmap2.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True)
#                 name = '/raft_seed_lrc_grad'
#                 # name = '/raft_seed'
#                 seed_disp = seed_growth(image1,image2,mkpts0,mkpts1,fmap1,fmap2,resize_shape,name) # [H,W]
#                 seed_disp = F.interpolate((seed_disp*ori_shape[1]/seed_disp.shape[1]).unsqueeze(0).unsqueeze(0), size=ori_shape, mode='nearest')
#                 seed_disp_numpy = seed_disp.squeeze(0).squeeze(0).numpy()
#                 print('time = %.2f' %(time.time() - start_time))
#             # # save seed result
#             numpy_dir = save_dir+name+'_epoch_{}.npy'.format(epoch-1)
#             png_dir = save_dir+name+'_epoch_{}.png'.format(epoch-1)
#             print('saving in {}'.format(png_dir))
#             np.save(numpy_dir, seed_disp_numpy)
#             toolkit.disp_vis(png_dir,seed_disp_numpy)
#             # # save raft result
#             numpy_dir = save_dir+'/raft_epoch_{}.npy'.format(epoch-1)
#             png_dir = save_dir+'/raft_epoch_{}.png'.format(epoch-1)
#             print('saving in {}'.format(png_dir))
#             np.save(numpy_dir, raft_disp)
#             toolkit.disp_vis(png_dir,raft_disp)
            
#             del fmap1,fmap2,flow_pr,raft_disp
#             torch.cuda.empty_cache()
#             time.sleep(5)
#             ## training ##
#             model.train()
            
#             optimizer.zero_grad()
        
#             # pad = toolkit.InputPadder(imageL.shape,16, mode = 'replicate')
#             # image1, image2 = pad.pad(imageL, imageR)
#             seed_disp = seed_disp.cuda()
#             mask = (seed_disp > 0)
#             mask.detach_()

#             assert model.training
#             flow_predictions = model(image1, image2, iters=args.train_iters)
#             for i in range(len(flow_predictions)):
#                 flow_predictions[i] = -padder.unpad(flow_predictions[i])
#                 flow_predictions[i] = F.interpolate((flow_predictions[i]*ori_shape[1]/flow_predictions[i].shape[-1]), size=ori_shape, mode='nearest')
#             assert model.training
            
#             loss, _ = sequence_loss(flow_predictions, seed_disp, mask)
#             print('learning_rate', optimizer.param_groups[0]['lr'])
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             scaler.step(optimizer)
#             scheduler.step()
#             scaler.update()
#             print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
#             total_train_loss += loss.item()
        
#         # evaluate epoch result
#         print('evaluate seed growth')
#         toolkit.evaluate((name+'_epoch_{}'.format(epoch-1))[1:],pre_mask=True)
#         print('evaluate raft')
#         toolkit.evaluate('raft_epoch_{}'.format(epoch-1),pre_mask=True)  
        
#         # save train loss 
#         train_loss_list.append(total_train_loss) 
#         np.save(os.path.join('trained/middlebury_seed','train_loss.npy'),np.array(train_loss_list))
#         len_x = np.arange(1,len(train_loss_list)+1)
#         plt.plot(len_x,np.array(train_loss_list),color='blue')
#         plt.savefig(os.path.join('trained/middlebury_seed','train_loss.png'))
#         plt.close()
        
#         # save model
#         savefilename = os.path.join('trained/middlebury_seed','epoch_'+str(epoch)+'.pth')
#         print('saving model in {}'.format(savefilename))
#         torch.save(model.state_dict(), savefilename)
#         print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(train_loader)))
        
        
#     print('full train time = %.2f HR' %((time.time() - start_full_time)/3600))

#     print("FINISHED TRAINING")

if __name__ == '__main__':
    # if args.data_type == 'middlebury':
    #     args.epochs = 100
    #     args.train_iters = 12
    #     n_images,data_loader = dataset.prepare_dataloader(['../datasets/middlebury/2016'],train=False,batch_size=1,mode='middlebury')
    #     print('number of middlebury2014 images: {}'.format(n_images))
    #     seed_train_middlebury(args,data_loader)
    if args.data_type == 'virtualroad':
        for road_mode in ['pt']:
            # for road_number in [5,2,0,6]:
            for road_number in [5,6]:
                if road_mode == 'pt' and road_number in [2,0]:
                    continue
                train_path_list,valid_path_list = toolkit.VirtualRoad_path_list(road_mode,road_number)
                if road_mode == 'ori' and int(road_number)==5:
                    train_path_list = ['../datasets/virtual_road/ori/t10_s106_r-10', '../datasets/virtual_road/ori/t10_s106_r0_bl01', '../datasets/virtual_road/ori/t10_s106_r10']
                elif road_mode == 'ori' and int(road_number)==6:
                    train_path_list = ['../datasets/virtual_road/ori/t10_s5_r-10', '../datasets/virtual_road/ori/t10_s5_r0_bl01', '../datasets/virtual_road/ori/t10_s5_r10']
                print('train_path_list: ',train_path_list,', valid_path_list: ',valid_path_list)
                n_Train,TrainImgLoader = dataset.prepare_dataloader(train_path_list,train=True,batch_size=2,mode='virtual_road',num_workers=4)
                n_Test,TestImgLoader = dataset.prepare_dataloader(valid_path_list,train=False,batch_size=5,mode='virtual_road',num_workers=1)
                print('number of train images: {}, number of valid images: {}'.format(n_Train,n_Test))
                train(args,TrainImgLoader,TestImgLoader,'{}_{}'.format(road_mode,road_number))
    elif args.data_type == 'apollo':
        train_path_list = ['../datasets/apollo']
        n_train,Train_loader, n_valid,valid_loader = dataset.prepare_dataloader(train_path_list,train=True,batch_size=4,mode='apollo',num_workers=4,resize=(300,1000),split=0.7)
        print('train_dataset:{},valid_dataset:{}'.format(n_train,n_valid))
        print('number of train images: {}, number of valid images: {}'.format(n_train,n_valid))
        train(args,Train_loader,valid_loader,'apollo')