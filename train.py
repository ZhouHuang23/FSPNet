
from torch.utils.data import DataLoader
import os
import builtins
import argparse
import torch
import torch.distributed as dist
import time
import FSPNet_model
import dataset
import loss


def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--path', type=str, help='path to train dataset')
    parser.add_argument('--pretrain', type=str, help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')
    
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args
                                         
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("args.rank = {}; args.gpu = {}".format(args.rank, args.gpu))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
       
    ### model ###
    net = FSPNet_model.Model(args.pretrain, img_size=384)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
            model_without_ddp = net.module
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
            model_without_ddp = net.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    encoder_param=[]
    decoer_param=[]
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    # optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}])
    
    ### resume training if necessary ###
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    ### Fine tuning for MoCA ###
    if args.ft_for_MoCA is not None:
        ckpt = torch.load(args.ft_for_MoCA, map_location='cpu')
        net.load_state_dict(ckpt)
        print("Fine tuning for MoCA, ckpt from: {}".format(args.ft_for_MoCA))

    
    ### data ###
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    Datasampler = torch.utils.data.distributed.DistributedSampler(Dataset, shuffle=True)
    Dataloader = DataLoader(Dataset, batch_size=args.batch_size_per_gpu, num_workers=args.batch_size_per_gpu, collate_fn=dataset.my_collate_fn, sampler=Datasampler, drop_last=True)
    
    # torch.backends.cudnn.benchmark = True
    
    ### main loop ###
    star_time=time.time()
    for curr_epoch in range(0, 201):
        
        if curr_epoch==100 or curr_epoch==150:
            for param_group in optimizer.param_groups:
                param_group['lr']= param_group['lr']*0.1
                print("Learning rate:", param_group['lr'])
        Datasampler.set_epoch(curr_epoch)
        net.train()
        running_loss_all, running_loss_m = 0., 0.
        count = 0
        for data in Dataloader:
            count += 1
            img, label = data['img'].cuda(args.rank), data['label'].cuda(args.rank)
            out = net(img)
            all_loss, m_loss = loss.multi_bce(out, label)
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            running_loss_all += all_loss.item()
            running_loss_m += m_loss.item()
            if count % 20 == 0 and args.rank == 0:
                print("Epoch:{}, Iter:{}, all_loss:{:.5f}, main_loss:{:.5f}".format(curr_epoch, count, running_loss_all / count, running_loss_m / count))
        if args.rank == 0 and curr_epoch % 2 == 0:
            ckpt_save_root = "/path_to_ckpt_save_root/ckpt_save"
            if not os.path.exists(ckpt_save_root):
                os.mkdir(ckpt_save_root)
            torch.save(net.state_dict(),
                       ckpt_save_root+"/model_{}_loss_{:.5f}.pth".format(curr_epoch, running_loss_m / count)
                       )


if __name__ == '__main__':
    args = parse_args()
    main(args)