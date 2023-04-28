import torch
import FSPNet_model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite

if __name__ =='__main__':
    batch_size = 1
    net = FSPNet_model.Model(None, img_size=384).cuda()

    ckpt=['FSPNet_best_0.01137.pth']

    Dirs=["/path_to_testset/TestDataset/CAMO",
          "/path_to_testset/TestDataset/CHAMELEON",
          "/path_to_testset/TestDataset/COD10K",
          "/path_to_testset/TestDataset/NC4K"]

    result_save_root="/path_to_save_root/results/"

    for m in ckpt:
        print(m)
        # pretrained_dict = torch.load("./ckpt/"+m)['model']
        ckpt_root="/path_to_ckpt_root/"
        ckpt_file="path_to_ckpt_files/"
        pretrained_dict = torch.load(ckpt_root+ckpt_file+m)

        net_dict = net.state_dict()
        pretrained_dict={k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict }
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists(result_save_root):
                os.mkdir(result_save_root)
            if not os.path.exists(os.path.join(result_save_root, Dir.split("/")[-1])):
                os.mkdir(os.path.join(result_save_root, Dir.split("/")[-1]))
            Dataset = dataset.TestDataset(Dir, 384)
            Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            count=0
            for data in Dataloader:
                count+=1
                img, label = data['img'].cuda(), data['label'].cuda()
                name = data['name'][0].split("/")[-1]
                with torch.no_grad():
                    out = net(img)[3]
                    # out = net(img)
                B,C,H,W = label.size()
                o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
                o =(o-o.min())/(o.max()-o.min()+1e-8)
                o = (o*255).astype(np.uint8)
                imwrite(result_save_root+Dir.split("/")[-1]+"/"+name, o)
    
    print("Test finished!")


