coding='utf-8'
import os
from net import RGBTSODnet
import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import cv2
import scipy.io as sio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SODloss(score1,score2,score_g,label):
    sal_loss2 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    label_g = F.interpolate(label, [25, 25], mode='bilinear', align_corners=True)
    sal_loss_g = F.binary_cross_entropy_with_logits(score_g, label_g, reduction='mean')
    return sal_loss2 + sal_loss3 + sal_loss_g
    
    
if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    
    # dataset
    img_root = '../../dataset6000'
    save_path = 'model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001 #2
    batch_size = 2
    epoch = 35
    lr_dec=[16,26]
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    device = torch.device("cuda:0")
    net = RGBTSODnet().cuda()
    net.load_pretrained_model()
    net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)
    iter_num = len(loader)
    net.train()
    for epochi in range(1, epoch + 1): 
    
        if epochi in lr_dec :
            lr=lr/10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)
            print(lr)
           
        prefetcher = DataPrefetcher(loader)
        rgb, t, label = prefetcher.next()
        r_sal_loss = 0
        net.zero_grad()
        i = 0
        while rgb is not None:
            i+=1
            result_r,result_t,result_g= net(rgb, t)
            sal_loss= SODloss(result_r,result_t,result_g,label)
            #sal_loss = F.binary_cross_entropy_with_logits(score, label, reduction='mean')
            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 10))
                #print(weit)
                #print(weit.max())
                r_sal_loss = 0
            rgb, t, label = prefetcher.next()
        torch.save(net.state_dict(), '%s/%d.pth' % (save_path,epochi))