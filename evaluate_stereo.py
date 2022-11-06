from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
sys.path.append("..")
import myself.toolkit as toolkit
from myself.seed_growth import seed_growth
from myself import dataset
sys.path.append("../LoFTR_master")
from LoFTR_master.lofrt_match import LoFTR_match
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--restore_ckpt', default='models/raftstereo-eth3d.pth', help="restore checkpoint")
parser.add_argument('--data_type', default='virtualroad', help="dataset type")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

# Architecure choices
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]= '4'

def load_model(args):
    
    model = torch.nn.DataParallel(RAFTStereo(args))
    # model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint from {}".format(args.restore_ckpt))
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")
    
    return model

def make_dataLoder(data_type): 
    if data_type == 'middlebury':
        len_, data_loader = dataset.prepare_dataloader(['../datasets/middlebury/2016'],train=False,batch_size=1,mode='middlebury',num_workers=0)

    elif 'virtualroad' in data_type:
        print('generate {} dataloader'.format('virtualroad'))
        road_mode = data_type.split('_')[-2]
        road_number = data_type.split('_')[-1]
        train_path_list,valid_path_list = toolkit.VirtualRoad_path_list(road_mode,int(road_number))
        # n_Train,Train_loader = dataset.prepare_dataloader(train_path_list,train=False,batch_size=1,mode='virtual_road',num_workers=1)
        n_Test,data_loader = dataset.prepare_dataloader(valid_path_list,train=False,batch_size=1,mode='virtual_road',num_workers=1)
    elif 'real_road' in data_type:
        paths_list = toolkit.real_road_path_list()
        len_, data_loader = dataset.prepare_dataloader(paths_list,train=False,batch_size=1,mode='real_road',num_workers=1)
    return data_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_virtualroad(data_loader, model, method, iters=32, mixed_prec=False):
    """ Peform validation using the virtualroad dataset """
    model.eval()
    aug_params = {}
    

    out_list, epe_list = [], []
    
    for (i,data) in enumerate(data_loader):
        image1 = data['left']
        image2 = data['right']
        gt_disp = data['disp'] 
        name = data['name'][0]
        image1_dir = data['left_dir'][0]
        image2_dir = data['right_dir'][0]
        save_dir = data['dir'][0]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        print('image1 {}, shape {}'.format(image1_dir,image1.shape))
        print('image2 {}, shape {}'.format(image2_dir,image2.shape))
        start_time = time.time()
        with autocast(enabled=mixed_prec):
            _, flow_pr,_,_ = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze()

        pred_disp = -flow_pr.detach().cpu().numpy()
        print('max_disp',np.max(pred_disp))
        print('time = %.2f' %(time.time() - start_time))

        # gt_disp = gt_disp[0].cpu().numpy()
        # toolkit.disp_vis('pre.png',pred_disp)
        # toolkit.disp_vis('gt.png',gt_disp)
        # exit()
        # wrong = np.zeros(pred_disp.shape)
        # diff = abs(pred_disp-gt_disp)
        # print(np.mean(pred_disp))
        # print(np.mean(gt_disp))
        # wrong[diff>3]=1
        # wrong[diff>gt_disp*0.05]=1
        # D1_all = np.sum(wrong)/760/1280
        # print(D1_all)
        # exit()

        save_dir = save_dir+'/{}'.format(method)
        numpy_dir = save_dir+'/numpy/'
        png_dir = save_dir+'/png/'
        if not os.path.exists(numpy_dir):
            os.makedirs(numpy_dir)
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        numpy_dir = numpy_dir+name+'.npy'
        png_dir = png_dir+name+'.png'
        
        print('mean',np.mean(pred_disp))
        
        print('save in {}'.format(save_dir))
        np.save(numpy_dir, pred_disp)
        toolkit.disp_vis(png_dir,pred_disp)
        torch.cuda.empty_cache()

    #     assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
    #     epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

    #     epe_flattened = epe.flatten()
    #     val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

    #     out = (epe_flattened > 2.0)
    #     image_out = out[val].float().mean().item()
    #     image_epe = epe_flattened[val].mean().item()
    #     logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
    #     epe_list.append(image_epe)
    #     out_list.append(image_out)

    # epe_list = np.array(epe_list)
    # out_list = np.array(out_list)

    # epe = np.mean(epe_list)
    # d1 = 100 * np.mean(out_list)

    # print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    # return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}
    
@torch.no_grad()
def validate_middlebury(data_loader, model, iters=12, mixed_prec=False):
    """ Peform validation using the virtualroad dataset """
    model.eval()
    
    for (i,data) in enumerate(data_loader):
        image1 = data['left']
        image2 = data['right']
        gt_disp = data['disp'] 
        name = data['name'][0]
        save_dir = data['dir'][0]

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        print('processing image {}, shape {}'.format(name,image1.shape))
        start_time = time.time()
        with autocast(enabled=mixed_prec):
            _, flow_pr,_,_ = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze()

        pred_disp = -flow_pr.detach().cpu().numpy()
        print('time = %.2f' %(time.time() - start_time))

        numpy_dir = save_dir+'/raft_no_train.npy'
        png_dir = save_dir+'/raft_no_train.png'
        print('save in {}'.format(save_dir))
        np.save(numpy_dir, pred_disp)
        toolkit.disp_vis(png_dir,pred_disp)
        torch.cuda.empty_cache()


@torch.no_grad()
def validate_middlebury_seed(data_loader, model, type ='general',iters=12, mixed_prec=False):
    """ Peform validation using the virtualroad dataset """
    model.eval()
    
    for (i,data) in enumerate(data_loader):
        image1 = data['left'].to('cuda')
        image2 = data['right'].to('cuda')
        name = data['name'][0]
        image1_dir = data['left_dir'][0]
        image2_dir = data['right_dir'][0]
        save_dir = data['dir'][0]
        print('processing image {}, shape {}'.format(name,image1.shape))
        start_time = time.time()
        
        ori_shape = (image1.shape[2],image1.shape[3])
        image1 = F.interpolate(image1, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)

        padder = toolkit.InputPadder(image1.shape, divide = 16, mode = 'replicate') # [N,H,W]
        image1, image2 = padder.pad(image1, image2)
        resize_shape = (512,1024)
        mkpts0,mkpts1 = LoFTR_match([image1_dir,image2_dir],'indoor',resize_shape)
        
        with autocast(enabled=mixed_prec):
            _, flow_pr,fmap1,fmap2 = model(image1, image2, iters=iters, test_mode=True)
        ratio = int(image1.shape[-1]/fmap1.shape[-1])
        new_size = (ratio * fmap1.shape[-2], ratio * fmap1.shape[-1])
        fmap1 = F.interpolate(fmap1, size=new_size, mode='bilinear', align_corners=True) # [1,C,H,W]
        fmap2 = F.interpolate(fmap2, size=new_size, mode='bilinear', align_corners=True)
        fmap1 = padder.unpad(fmap1[0]) # [C,H,W]
        fmap2 = padder.unpad(fmap2[0])
        fmap1 = F.interpolate(fmap1.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True) # [1,C,H,W]
        fmap2 = F.interpolate(fmap2.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True)
        
        name = '/raft_seed_lrc_grad'
        name = '/raft_seed'
        name = '/raft_seed_lrc'
        name = '/raft_seed_grad'
        
        pred_disp = seed_growth(image1,image2,mkpts0,mkpts1,fmap1,fmap2,resize_shape,name) # [H,W]
        pred_disp = F.interpolate((pred_disp*ori_shape[1]/pred_disp.shape[1]).unsqueeze(0).unsqueeze(0), size=ori_shape, mode='nearest')
        # pred_disp = F.interpolate((pred_disp*ori_shape[1]/pred_disp.shape[1]).unsqueeze(0).unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=True)
        pred_disp = pred_disp.squeeze(0).squeeze(0).numpy()
        
        print('time = %.2f' %(time.time() - start_time))

        numpy_dir = save_dir+name+'.npy'
        png_dir = save_dir+name+'.png'
        print('save in {}'.format(save_dir + name))
        np.save(numpy_dir, pred_disp)
        toolkit.disp_vis(png_dir,pred_disp)
        torch.cuda.empty_cache()
    toolkit.evaluate(name[1:],pre_mask=True)


@torch.no_grad()
def validate_seed(data_loader, model, method, iters=12, mixed_prec=False):
    """ Peform validation using the virtualroad dataset """
    model.eval()
    
    for (i,data) in enumerate(data_loader):
        image1 = data['left'].to('cuda')
        image2 = data['right'].to('cuda')
        name = data['name'][0]
        image1_dir = data['left_dir'][0]
        image2_dir = data['right_dir'][0]
        save_dir = data['dir'][0]
        print('processing image {}, shape {}'.format(name,image1.shape))
        start_time = time.time()
        
        ori_shape = (image1.shape[2],image1.shape[3])
        image1 = F.interpolate(image1, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, size=(int(ori_shape[0]/2),int(ori_shape[1]/2)), mode='bilinear', align_corners=True)

        padder = toolkit.InputPadder(image1.shape, divide = 16, mode = 'replicate') # [N,H,W]
        image1, image2 = padder.pad(image1, image2)
        resize_shape = (512,1024)
        mkpts0,mkpts1 = LoFTR_match([image1_dir,image2_dir],'indoor',resize_shape)
        
        with autocast(enabled=mixed_prec):
            _, flow_pr,fmap1,fmap2 = model(image1, image2, iters=iters, test_mode=True)
        ratio = int(image1.shape[-1]/fmap1.shape[-1])
        new_size = (ratio * fmap1.shape[-2], ratio * fmap1.shape[-1])
        fmap1 = F.interpolate(fmap1, size=new_size, mode='bilinear', align_corners=True) # [1,C,H,W]
        fmap2 = F.interpolate(fmap2, size=new_size, mode='bilinear', align_corners=True)
        fmap1 = padder.unpad(fmap1[0]) # [C,H,W]
        fmap2 = padder.unpad(fmap2[0])
        fmap1 = F.interpolate(fmap1.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True) # [1,C,H,W]
        fmap2 = F.interpolate(fmap2.unsqueeze(0), size=resize_shape, mode='bilinear', align_corners=True)
        
        pred_disp = seed_growth(image1,image2,mkpts0,mkpts1,fmap1,fmap2,resize_shape,method) # [H,W]
        pred_disp = F.interpolate((pred_disp*ori_shape[1]/pred_disp.shape[1]).unsqueeze(0).unsqueeze(0), size=ori_shape, mode='nearest')
        # pred_disp = F.interpolate((pred_disp*ori_shape[1]/pred_disp.shape[1]).unsqueeze(0).unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=True)
        pred_disp = pred_disp.squeeze(0).squeeze(0).numpy()
        
        print('time = %.2f' %(time.time() - start_time))

        numpy_dir = save_dir+'/{}/numpy/'.format(method)
        png_dir = save_dir+'/{}/png/'.format(method)
        if not os.path.exists(numpy_dir):
            os.makedirs(numpy_dir)
        if not os.path.exists(png_dir):
            os.makedirs(png_dir)
        numpy_dir = numpy_dir+name+'.npy'
        png_dir = png_dir+name+'.png'
        
        print('save in {}'.format(save_dir + numpy_dir))
        np.save(numpy_dir, pred_disp)
        toolkit.disp_vis(png_dir,pred_disp)
        torch.cuda.empty_cache()
    # toolkit.evaluate(name[1:],pre_mask=True)

if __name__ == '__main__':
    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.data_type == 'middlebury':
        model = load_model(args)
        data_loader = make_dataLoder(args.data_type)
        with torch.no_grad():
            # validate_middlebury(data_loader,model)
            validate_middlebury_seed(data_loader,model)
    elif 'virtualroad' in args.data_type:
        # for mode in ['pt']:
        for mode in ['ori']:
            for number in ['2','5','0','6']:
            # for number in ['5']:
            # for number in ['5']:
                if mode == 'pt' and number in ['2','0']:
                    continue
                # args.restore_ckpt = 'trained/{}_{}/finetune_10_56.pth'.format(mode,number)
                # args.restore_ckpt = 'trained/{}_{}/finetune_2_56.pth'.format(mode,number)
                args.restore_ckpt = 'trained/{}_{}/finetune_6.pth'.format(mode,number)
                model = load_model(args)
                data_loader = make_dataLoder(args.data_type+'_{}_{}'.format(mode,number))
                with torch.no_grad():
                    
                    if 'seed' in args.data_type:
                        method = 'raft_seed_pt_only'
                        method = 'raft_seed_sub_lrc_pt'
                        method = 'raft_seed_sub_lrc'
                        validate_seed(data_loader,model,method)
                    else:
                        method = 'raft_finetune_best'
                        method = 'raft_finetune_56'
                        method = 'raft_no_train'
                        method = 'raft_finetune'
                        validate_virtualroad(data_loader,model,method)
    elif 'real_road' in args.data_type:
        # args.restore_ckpt = 'trained/pt_6/finetune_1.pth'
        model = load_model(args)
        data_loader = make_dataLoder(args.data_type)
        # method = 'raft_finetune'
        
        with torch.no_grad():
            if 'seed' in args.data_type:
                method = 'raft_seed_sub_lrc'
                method = 'raft_seed_sub_lrc_pt'
                validate_seed(data_loader,model,method)
            else:
                method = 'raft_no_train'
                validate_virtualroad(data_loader,model,method)
                
