# RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion
This project provides the codes and results for 'RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion.'

## Overview
![image](https://github.com/xanxuso/RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion/blob/main/network.png)

## Code
### For train:
download pretrained vgg model from [link](https://pan.baidu.com/s/1DDHhmjau01Oo775mi1wdgw),code：0000; put it in `model` directory.

change path to train data in `train.py`.

use `python train.py` to begin training.
### For resultfusion:
use `python resultFusion.py` to fuse saliency maps from RGB and T modalities, there are some examples in `asserts`

## Experimental Results
Result of VT5000 dataset: [link](https://pan.baidu.com/s/1jn6Y9vi7qhnTIpHiW8anWA)，&emsp;code：1234

Result of VT1000 dataset: [link](https://pan.baidu.com/s/1jex2q55VZeSalOYWCtC4BQ)，&emsp;code：4321
