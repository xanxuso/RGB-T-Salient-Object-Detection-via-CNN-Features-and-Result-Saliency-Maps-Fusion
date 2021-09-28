coding='utf-8'
import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from net import RGBTSODnet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    model_path='path to trained checkpoint'
    out_path = 'path to save results'
    data  = Data(root='path to test data',mode='test')
    # data  = Data(root='../campusdataset/night',mode='test')
    loader = DataLoader(data, batch_size=1,shuffle=False)
    net = RGBTSODnet().cuda()
    device = torch.device("cuda:0")
    net = torch.nn.DataParallel(net)
    net.to(device)
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for rgb, t, _ , (H, W), name in loader:
            result_r, result_t, result_g = net(rgb.cuda().float(), t.cuda().float())

            result_r = F.interpolate(result_r, size=(H, W), mode='bilinear')
            pred1 = np.squeeze(torch.sigmoid(result_r).cpu().data.numpy())
            pred1 = 255 * pred1
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '_1.png'), pred1)

            result_t = F.interpolate(result_t, size=(H, W), mode='bilinear')
            pred2 = np.squeeze(torch.sigmoid(result_t).cpu().data.numpy())
            pred2 = 255 * pred2
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '_2.png'), pred2)

            result_g = F.interpolate(result_g, size=(H, W), mode='bilinear')
            pred3 = np.squeeze(torch.sigmoid(result_g).cpu().data.numpy())
            pred3 = 255 * pred3
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '_g.png'), pred3)            
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))

